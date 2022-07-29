use crate::state_registry::{StateInRegistry, StateInformation};
use dypdl::variable_type::Numeric;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

/// A trait representing a search node.
pub trait DPSearchNode<T: Numeric>: StateInformation<T> {
    /// Returns the parent node.
    fn parent(&self) -> Option<Self>;

    /// Returns the transition to reach this node.
    fn operator(&self) -> Option<Rc<dypdl::Transition>>;
}

/// Returns the transition from the initial node to reach this node.
pub fn trace_transitions<T: Numeric, N: DPSearchNode<T>>(mut node: N) -> Vec<dypdl::Transition> {
    let mut result = Vec::new();
    while let (Some(parent), Some(operator)) = (node.parent(), node.operator()) {
        result.push(operator.as_ref().clone());
        node = parent;
    }
    result.reverse();
    result
}

/// Search node.
///
/// Nodes are totally ordered by their costs.
#[derive(Debug, Default)]
pub struct SearchNode<T: Numeric> {
    // State.
    pub state: StateInRegistry,
    // Cost to reach this node.
    pub cost: T,
    /// Already expanded or not.
    pub closed: RefCell<bool>,
    /// Parent node.
    pub parent: Option<Rc<SearchNode<T>>>,
    /// Transition to reach this node.
    pub operator: Option<Rc<dypdl::Transition>>,
}

impl<T: Numeric + PartialOrd> PartialEq for SearchNode<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl<T: Numeric + Ord> Eq for SearchNode<T> {}

impl<T: Numeric + Ord> Ord for SearchNode<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl<T: Numeric + Ord> PartialOrd for SearchNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Numeric> StateInformation<T> for Rc<SearchNode<T>> {
    #[inline]
    fn state(&self) -> &StateInRegistry {
        &self.state
    }

    #[inline]
    fn cost(&self) -> T {
        self.cost
    }
}

impl<T: Numeric> DPSearchNode<T> for Rc<SearchNode<T>> {
    #[inline]
    fn parent(&self) -> Option<Self> {
        self.parent.as_ref().cloned()
    }

    #[inline]
    fn operator(&self) -> Option<Rc<dypdl::Transition>> {
        self.operator.as_ref().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashable_state::HashableSignatureVariables;

    #[test]
    fn search_node_getter() {
        let node = Rc::new(SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        });
        assert_eq!(node.state(), &node.state);
        assert_eq!(node.cost(), 0);
        assert!(node.parent().is_none());
        assert!(node.operator().is_none());
    }

    #[test]
    fn search_node_cmp() {
        let node1 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![1, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        let node2 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 0,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert_eq!(node1, node2);
        let node2 = SearchNode {
            state: StateInRegistry {
                signature_variables: Rc::new(HashableSignatureVariables {
                    integer_variables: vec![4, 2, 3],
                    ..Default::default()
                }),
                ..Default::default()
            },
            cost: 2,
            closed: RefCell::new(false),
            parent: None,
            operator: None,
        };
        assert!(node1 < node2)
    }

    #[test]
    fn trace_transitions_test() {
        let op1 = dypdl::Transition {
            name: String::from("op1"),
            ..Default::default()
        };
        let op2 = dypdl::Transition {
            name: String::from("op2"),
            ..Default::default()
        };
        let node1 = Rc::new(SearchNode {
            cost: 0,
            ..Default::default()
        });
        let node2 = Rc::new(SearchNode {
            operator: Some(Rc::new(op1.clone())),
            parent: Some(node1),
            cost: 1,
            ..Default::default()
        });
        let node3 = Rc::new(SearchNode {
            operator: Some(Rc::new(op2.clone())),
            parent: Some(node2),
            cost: 1,
            ..Default::default()
        });
        let result = trace_transitions(node3);
        let expected = vec![op1, op2];
        assert_eq!(result, expected);
    }
}