use super::f_evaluator_type::FEvaluatorType;
use super::search_algorithm::data_structure::state_registry::StateInRegistry;
use super::search_algorithm::data_structure::{FNode, SuccessorGenerator};
use super::search_algorithm::util::{ForwardSearchParameters, Parameters};
use super::search_algorithm::Dbdfs;
use super::search_algorithm::Search;
use dypdl::variable_type;
use std::fmt;
use std::rc::Rc;
use std::str;

/// Creates a Discrepancy-Bounded Depth-First Search (DBDFS) solver using the dual bound as a heuristic function.
///
/// This solver uses forward search based on the shortest path problem.
/// It only works with problems where the cost expressions are in the form of `cost + w`, `cost * w`, `max(cost, w)`, or `min(cost, w)`
/// where `cost` is `IntegerExpression::Cost`or `ContinuousExpression::Cost` and `w` is a numeric expression independent of `cost`.
/// `f_evaluator_type` must be specified appropriately according to the cost expressions.
///
/// `width` is the width of the discrepancy to search.
///
/// # References
///
/// Ryo Kuroiwa and J. Christopher Beck. "Solving Domain-Independent Dynamic Programming with Anytime Heuristic Search,"
/// Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS), 2023.
///
/// J. Christopher Beck. "Discrepancy-Bounded Depth First Search,"
/// Second International Workshop on Integration of AI and OR Technologies in Constraint Programming for Combinatorial Optimization Problems (CPAIOR), 2000.
///
/// # Examples
///
/// ```
/// use dypdl::prelude::*;
/// use dypdl_heuristic_search::{FEvaluatorType, create_dual_bound_dbdfs};
/// use dypdl_heuristic_search::search_algorithm::util::Parameters;
/// use std::rc::Rc;
///
/// let mut model = Model::default();
/// let variable = model.add_integer_variable("variable", 0).unwrap();
/// model.add_base_case(
///     vec![Condition::comparison_i(ComparisonOperator::Ge, variable, 1)]
/// ).unwrap();
/// let mut increment = Transition::new("increment");
/// increment.set_cost(IntegerExpression::Cost + 1);
/// increment.add_effect(variable, variable + 1).unwrap();
/// model.add_forward_transition(increment.clone()).unwrap();
/// model.add_dual_bound(IntegerExpression::from(0)).unwrap();
///
/// let model = Rc::new(model);
/// let parameters = Parameters::default();
/// let f_evaluator_type = FEvaluatorType::Plus;
///
/// let mut solver = create_dual_bound_dbdfs(model, parameters, 1, f_evaluator_type, None);
/// let solution = solver.search().unwrap();
/// assert_eq!(solution.cost, Some(1));
/// assert_eq!(solution.transitions, vec![increment]);
/// assert!(!solution.is_infeasible);
/// ```
pub fn create_dual_bound_dbdfs<T>(
    model: Rc<dypdl::Model>,
    parameters: Parameters<T>,
    width: usize,
    f_evaluator_type: FEvaluatorType,
    initial_registry_capacity: Option<usize>,
) -> Box<dyn Search<T>>
where
    T: variable_type::Numeric + fmt::Display + Ord + 'static,
    <T as str::FromStr>::Err: fmt::Debug,
{
    let generator = SuccessorGenerator::from_model(model.clone(), false);
    let parameters = ForwardSearchParameters {
        generator,
        parameters,
        initial_registry_capacity,
    };
    let h_model = model.clone();
    let h_evaluator =
        move |state: &StateInRegistry| Some(h_model.eval_dual_bound(state).unwrap_or_else(T::zero));
    let (f_pruning, f_evaluator_type) = if model.has_dual_bounds() {
        (true, f_evaluator_type)
    } else {
        (false, FEvaluatorType::Plus)
    };
    let f_evaluator = move |g, h, _: &StateInRegistry| f_evaluator_type.eval(g, h);

    Box::new(Dbdfs::<_, FNode<_>, _, _>::new(
        model,
        h_evaluator,
        f_evaluator,
        f_pruning,
        width,
        parameters,
    ))
}
