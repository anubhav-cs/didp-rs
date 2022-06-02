use crate::bfs_node::BFSNode;
use crate::evaluator;
use crate::search_node::trace_transitions;
use crate::solver;
use crate::state_registry::{StateInRegistry, StateInformation, StateRegistry};
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections;
use std::fmt;
use std::rc::Rc;

pub fn forward_bfs<T, H, F>(
    model: &didp_parser::Model<T>,
    generator: SuccessorGenerator<didp_parser::Transition<T>>,
    h_evaluator: H,
    f_evaluator: F,
    g_bound: Option<T>,
    registry_capacity: Option<usize>,
) -> solver::Solution<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    H: evaluator::Evaluator<T>,
    F: Fn(T, T, &StateInRegistry, &didp_parser::Model<T>) -> T,
{
    let mut open = collections::BinaryHeap::new();
    let mut registry = StateRegistry::new(model);
    if let Some(capacity) = registry_capacity {
        registry.reserve(capacity);
    }

    let g = T::zero();
    let initial_state = StateInRegistry::new(&model.target);
    let h = h_evaluator.eval(&initial_state, model)?;
    println!("Initial h = {}", h);
    let f = f_evaluator(g, h, &initial_state, model);
    let constructor = |state: StateInRegistry, g: T, _: Option<&Rc<BFSNode<T>>>| {
        Some(Rc::new(BFSNode {
            state,
            g,
            h: RefCell::new(Some(h)),
            f: RefCell::new(Some(f)),
            ..Default::default()
        }))
    };
    let initial_node = match registry.insert(initial_state, g, constructor) {
        Some((node, _)) => node,
        None => return None,
    };
    open.push(Reverse(initial_node));
    let mut expanded = 0;
    let mut f_max = f;

    while let Some(Reverse(node)) = open.pop() {
        if *node.closed.borrow() {
            continue;
        }
        *node.closed.borrow_mut() = true;
        expanded += 1;
        let f = node.f.borrow().unwrap();
        if f > f_max {
            f_max = f;
            println!("f = {}, expanded: {}", f, expanded);
        }
        if model.get_base_cost(node.state()).is_some() {
            println!("Expanded: {}", expanded);
            return Some((node.g, trace_transitions(node)));
        }
        for transition in generator.applicable_transitions(node.state()) {
            let g = transition.eval_cost(node.g, node.state(), &model.table_registry);
            if g_bound.is_some() && g >= g_bound.unwrap() {
                continue;
            }
            let state = transition.apply(node.state(), &model.table_registry);
            if model.check_constraints(&state) {
                let constructor = |state: StateInRegistry, g: T, other: Option<&Rc<BFSNode<T>>>| {
                    // use a cached h-value
                    let h = if let Some(other) = other {
                        match *other.h.borrow() {
                            None => return None,
                            h => h,
                        }
                    } else {
                        None
                    };
                    Some(Rc::new(BFSNode {
                        state,
                        g,
                        h: RefCell::new(h),
                        f: RefCell::new(None),
                        parent: Some(node.clone()),
                        operator: Some(transition),
                        closed: RefCell::new(false),
                    }))
                };
                if let Some((successor, dominated)) = registry.insert(state, g, constructor) {
                    if let Some(dominated) = dominated {
                        if !*dominated.closed.borrow() {
                            *dominated.closed.borrow_mut() = true;
                        }
                    }
                    let h = match *successor.h.borrow() {
                        None => h_evaluator.eval(successor.state(), model),
                        h => h,
                    };
                    if let Some(h) = h {
                        let f = f_evaluator(g, h, successor.state(), model);
                        if g_bound.is_some() && f >= g_bound.unwrap() {
                            continue;
                        }
                        *successor.h.borrow_mut() = Some(h);
                        *successor.f.borrow_mut() = Some(f);
                        open.push(Reverse(successor));
                    }
                }
            }
        }
    }
    println!("Expanded: {}", expanded);
    None
}
