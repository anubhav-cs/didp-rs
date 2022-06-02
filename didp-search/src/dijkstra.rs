use crate::search_node::{trace_transitions, SearchNode};
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use std::str;

#[derive(Default)]
pub struct Dijkstra<T> {
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for Dijkstra<T>
where
    T: variable::Numeric + Ord + fmt::Display,
{
    #[inline]
    fn set_primal_bound(&mut self, primal_bound: Option<T>) {
        self.primal_bound = primal_bound
    }

    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = SuccessorGenerator::<didp_parser::Transition<T>>::new(model, false);
        Ok(dijkstra(
            model,
            generator,
            self.primal_bound,
            self.registry_capacity,
        ))
    }
}

impl<T: variable::Numeric> Dijkstra<T>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    pub fn new(config: &yaml_rust::Yaml) -> Result<Dijkstra<T>, Box<dyn Error>> {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            yaml_rust::Yaml::Null => return Ok(Dijkstra::default()),
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                ))
                .into())
            }
        };
        let primal_bound = match map.get(&yaml_rust::Yaml::from_str("primal_bound")) {
            Some(yaml_rust::Yaml::Integer(value)) => {
                Some(T::from_integer(*value as variable::Integer))
            }
            Some(yaml_rust::Yaml::Real(value)) => Some(value.parse().map_err(|e| {
                solver::ConfigErr::new(format!("could not parse {} as a number: {:?}", value, e))
            })?),
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let registry_capacity = match map.get(&yaml_rust::Yaml::from_str("registry_capacity")) {
            Some(yaml_rust::Yaml::Integer(value)) => Some(*value as usize),
            None => Some(1000000),
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        Ok(Dijkstra {
            primal_bound,
            registry_capacity,
        })
    }
}

pub fn dijkstra<T>(
    model: &didp_parser::Model<T>,
    generator: SuccessorGenerator<didp_parser::Transition<T>>,
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
{
    let mut open = collections::BinaryHeap::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let cost = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let constructor = |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
        Some(Rc::new(SearchNode {
            state,
            cost,
            ..Default::default()
        }))
    };
    let initial_node = match registry.insert(initial_state, cost, constructor) {
        Some((node, _)) => node,
        None => return None,
    };
    open.push(Reverse(initial_node));
    let mut expanded = 0;
    let mut cost_max = T::zero();

    while let Some(Reverse(node)) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        if node.cost() > cost_max {
            cost_max = node.cost();
            println!("cost = {}, expanded: {}", cost_max, expanded);
        }
        if model.get_base_cost(node.state()).is_some() {
            println!("Expanded: {}", expanded);
            return Some((node.cost(), trace_transitions(node)));
        }
        for transition in generator.applicable_transitions(node.state()) {
            let cost = transition.eval_cost(node.cost(), node.state(), &model.table_registry);
            if primal_bound.is_some() && cost >= primal_bound.unwrap() {
                continue;
            }
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor =
                    |state: StateInRegistry, cost: T, _: Option<&Rc<SearchNode<T>>>| {
                        Some(Rc::new(SearchNode {
                            state,
                            cost,
                            parent: Some(node.clone()),
                            operator: Some(transition.clone()),
                            closed: RefCell::new(false),
                        }))
                    };
                if let Some((successor, dominated)) = registry.insert(state, cost, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    open.push(Reverse(successor));
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    None
}
