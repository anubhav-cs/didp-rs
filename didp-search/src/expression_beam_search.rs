use super::expression_astar::FEvaluatorType;
use crate::expression_evaluator::ExpressionEvaluator;
use crate::forward_beam_search;
use crate::forward_bfs;
use crate::search_node::StateForSearchNode;
use crate::search_node::TransitionWithG;
use crate::solver;
use crate::successor_generator::SuccessorGenerator;
use didp_parser::variable;
use rustc_hash::FxHashMap;
use std::cmp;
use std::error::Error;
use std::fmt;
use std::str;

pub struct ExpressionBeamSearch<T: variable::Numeric> {
    g_expressions: Option<FxHashMap<String, String>>,
    h_expression: Option<String>,
    f_evaluator_type: FEvaluatorType,
    beams: Vec<usize>,
    maximize: bool,
    primal_bound: Option<T>,
    registry_capacity: Option<usize>,
}

impl<T> solver::Solver<T> for ExpressionBeamSearch<T>
where
    T: variable::Numeric + Ord + fmt::Display,
    <T as str::FromStr>::Err: fmt::Debug,
{
    #[inline]
    fn set_primal_bound(&mut self, bound: Option<T>) {
        self.primal_bound = bound;
    }

    fn solve(
        &mut self,
        model: &didp_parser::Model<T>,
    ) -> Result<solver::Solution<T>, Box<dyn Error>> {
        let generator = if let Some(g_expressions) = self.g_expressions.as_ref() {
            SuccessorGenerator::with_expressions(&model, false, g_expressions)?
        } else {
            SuccessorGenerator::<TransitionWithG<T>>::new(&model, false)
        };
        let h_evaluator = if let Some(h_expression) = self.h_expression.as_ref() {
            ExpressionEvaluator::new(h_expression.clone(), &model)?
        } else {
            ExpressionEvaluator::default()
        };
        let solution = match self.f_evaluator_type {
            FEvaluatorType::Plus => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| g + h);
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &evaluators,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Max => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::max(g, h)
                    });
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &evaluators,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Min => {
                let f_evaluator =
                    Box::new(|g, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| {
                        cmp::min(g, h)
                    });
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &evaluators,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
            FEvaluatorType::Overwrite => {
                let f_evaluator =
                    Box::new(|_, h, _: &StateForSearchNode, _: &didp_parser::Model<T>| h);
                let evaluators = forward_bfs::BFSEvaluators {
                    generator,
                    h_evaluator,
                    f_evaluator,
                };
                forward_beam_search::iterative_forward_beam_search(
                    model,
                    &evaluators,
                    &self.beams,
                    self.maximize,
                    self.primal_bound,
                    self.registry_capacity,
                )
            }
        };
        Ok(solution)
    }
}

impl<T: variable::Numeric + Ord> ExpressionBeamSearch<T> {
    pub fn new(config: &yaml_rust::Yaml) -> Result<ExpressionBeamSearch<T>, Box<dyn Error>>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = match config {
            yaml_rust::Yaml::Hash(map) => map,
            _ => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    config
                ))
                .into())
            }
        };
        let g_expressions = match map.get(&yaml_rust::Yaml::from_str("g")) {
            Some(yaml_rust::Yaml::Hash(map)) => {
                let mut g_expressions = FxHashMap::default();
                g_expressions.reserve(map.len());
                for (key, value) in map.iter() {
                    match (key, value) {
                        (yaml_rust::Yaml::String(key), yaml_rust::Yaml::String(value)) => {
                            g_expressions.insert(key.clone(), value.clone());
                        }
                        _ => {
                            return Err(solver::ConfigErr::new(format!(
                                "expected (String, String), but found (`{:?}`, `{:?}`)",
                                key, value
                            ))
                            .into())
                        }
                    }
                }
                Some(g_expressions)
            }
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Hash, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let h_expression = match map.get(&yaml_rust::Yaml::from_str("h")) {
            Some(yaml_rust::Yaml::String(value)) => Some(value.clone()),
            None => None,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let f_evaluator_type = match map.get(&yaml_rust::Yaml::from_str("f")) {
            Some(yaml_rust::Yaml::String(string)) => match &string[..] {
                "+" => FEvaluatorType::Plus,
                "max" => FEvaluatorType::Max,
                "min" => FEvaluatorType::Min,
                "h" => FEvaluatorType::Overwrite,
                op => {
                    return Err(solver::ConfigErr::new(format!(
                        "unexpected operator for f function `{}`",
                        op
                    ))
                    .into())
                }
            },
            None => FEvaluatorType::Plus,
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected String, but found `{:?}`",
                    value
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
        let beams = match map.get(&yaml_rust::Yaml::from_str("beam")) {
            Some(yaml_rust::Yaml::Integer(value)) => vec![*value as usize],
            Some(yaml_rust::Yaml::Array(array)) => {
                let mut beams = Vec::new();
                for v in array {
                    match v {
                        yaml_rust::Yaml::Integer(value) => {
                            beams.push(*value as usize);
                        }
                        value => {
                            return Err(solver::ConfigErr::new(format!(
                                "expected Integer or Array, but found `{:?}`",
                                value
                            ))
                            .into())
                        }
                    }
                }
                beams
            }
            value => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Integer or Array, but found `{:?}`",
                    value
                ))
                .into())
            }
        };
        let maximize = match map.get(&yaml_rust::Yaml::from_str("maximize")) {
            Some(yaml_rust::Yaml::Boolean(value)) => *value,
            Some(value) => {
                return Err(solver::ConfigErr::new(format!(
                    "expected Boolean, but found `{:?}`",
                    value
                ))
                .into())
            }
            None => false,
        };
        Ok(ExpressionBeamSearch {
            g_expressions,
            h_expression,
            f_evaluator_type,
            primal_bound,
            beams,
            maximize,
            registry_capacity,
        })
    }
}
