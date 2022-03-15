use super::numeric_parser;
use super::set_parser;
use super::util;
use super::util::ParseErr;
use crate::expression::{ComparisonOperator, Condition, SetCondition};
use crate::function_registry;
use crate::state;
use crate::variable;
use std::collections;
use std::fmt;
use std::str;

pub fn parse_expression<'a, 'b, 'c, T: variable::Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b function_registry::FunctionRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (token, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &token[..] {
        "(" => parse_operation(rest, metadata, registry, parameters),
        _ => Err(ParseErr::Reason(format!("unexpected token: `{}`", token))),
    }
}

fn parse_operation<'a, 'b, 'c, T: variable::Numeric>(
    tokens: &'a [String],
    metadata: &'b state::StateMetadata,
    registry: &'b function_registry::FunctionRegistry<T>,
    parameters: &'c collections::HashMap<String, usize>,
) -> Result<(Condition<'b, T>, &'a [String]), ParseErr>
where
    <T as str::FromStr>::Err: fmt::Debug,
{
    let (name, rest) = tokens
        .split_first()
        .ok_or_else(|| ParseErr::Reason("could not get token".to_string()))?;
    match &name[..] {
        "not" => {
            let (condition, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Not(Box::new(condition)), rest))
        }
        "and" => {
            let (x, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::And(Box::new(x), Box::new(y)), rest))
        }
        "or" => {
            let (x, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Or(Box::new(x), Box::new(y)), rest))
        }
        "=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Eq, x, y), rest))
        }
        "!=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ne, x, y), rest))
        }
        ">=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Ge, x, y), rest))
        }
        ">" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Gt, x, y), rest))
        }
        "<=" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Le, x, y), rest))
        }
        "<" => {
            let (x, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let (y, rest) = numeric_parser::parse_expression(rest, metadata, registry, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Comparison(ComparisonOperator::Lt, x, y), rest))
        }
        "is_in" => {
            let (e, rest) = set_parser::parse_element_expression(rest, metadata, parameters)?;
            let (s, rest) = set_parser::parse_set_expression(rest, metadata, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsIn(e, s)), rest))
        }
        "is_subset" => {
            let (x, rest) = set_parser::parse_set_expression(rest, metadata, parameters)?;
            let (y, rest) = set_parser::parse_set_expression(rest, metadata, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsSubset(x, y)), rest))
        }
        "is_empty" => {
            let (s, rest) = set_parser::parse_set_expression(rest, metadata, parameters)?;
            let rest = util::parse_closing(rest)?;
            Ok((Condition::Set(SetCondition::IsEmpty(s)), rest))
        }
        _ => Err(ParseErr::Reason(format!("no such operator: `{}`", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::*;
    use crate::numeric_function;
    use std::collections::HashMap;

    fn generate_metadata() -> state::StateMetadata {
        let object_names = vec!["object".to_string()];
        let object_numbers = vec![10];
        let mut name_to_object = HashMap::new();
        name_to_object.insert("object".to_string(), 0);

        let set_variable_names = vec![
            "s0".to_string(),
            "s1".to_string(),
            "s2".to_string(),
            "s3".to_string(),
        ];
        let mut name_to_set_variable = HashMap::new();
        name_to_set_variable.insert("s0".to_string(), 0);
        name_to_set_variable.insert("s1".to_string(), 1);
        name_to_set_variable.insert("s2".to_string(), 2);
        name_to_set_variable.insert("s3".to_string(), 3);
        let set_variable_to_object = vec![0, 0, 0, 0];

        let permutation_variable_names = vec![
            "p0".to_string(),
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let mut name_to_permutation_variable = HashMap::new();
        name_to_permutation_variable.insert("p0".to_string(), 0);
        name_to_permutation_variable.insert("p1".to_string(), 1);
        name_to_permutation_variable.insert("p2".to_string(), 2);
        name_to_permutation_variable.insert("p3".to_string(), 3);
        let permutation_variable_to_object = vec![0, 0, 0, 0];

        let element_variable_names = vec![
            "e0".to_string(),
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
        ];
        let mut name_to_element_variable = HashMap::new();
        name_to_element_variable.insert("e0".to_string(), 0);
        name_to_element_variable.insert("e1".to_string(), 1);
        name_to_element_variable.insert("e2".to_string(), 2);
        name_to_element_variable.insert("e3".to_string(), 3);
        let element_variable_to_object = vec![0, 0, 0, 0];

        let numeric_variable_names = vec![
            "n0".to_string(),
            "n1".to_string(),
            "n2".to_string(),
            "n3".to_string(),
        ];
        let mut name_to_numeric_variable = HashMap::new();
        name_to_numeric_variable.insert("n0".to_string(), 0);
        name_to_numeric_variable.insert("n1".to_string(), 1);
        name_to_numeric_variable.insert("n2".to_string(), 2);
        name_to_numeric_variable.insert("n3".to_string(), 3);

        let resource_variable_names = vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ];
        let mut name_to_resource_variable = HashMap::new();
        name_to_resource_variable.insert("r0".to_string(), 0);
        name_to_resource_variable.insert("r1".to_string(), 1);
        name_to_resource_variable.insert("r2".to_string(), 2);
        name_to_resource_variable.insert("r3".to_string(), 3);

        state::StateMetadata {
            object_names,
            name_to_object,
            object_numbers,
            set_variable_names,
            name_to_set_variable,
            set_variable_to_object,
            permutation_variable_names,
            name_to_permutation_variable,
            permutation_variable_to_object,
            element_variable_names,
            name_to_element_variable,
            element_variable_to_object,
            numeric_variable_names,
            name_to_numeric_variable,
            resource_variable_names,
            name_to_resource_variable,
            less_is_better: vec![false, false, true, false],
        }
    }

    fn generate_parameters() -> collections::HashMap<String, usize> {
        let mut parameters = collections::HashMap::new();
        parameters.insert("param".to_string(), 0);
        parameters
    }

    fn generate_registry() -> function_registry::FunctionRegistry<variable::IntegerVariable> {
        let mut functions_1d = HashMap::new();
        let f1 = numeric_function::NumericFunction1D::new(Vec::new());
        functions_1d.insert("f1".to_string(), f1);

        let mut functions_2d = HashMap::new();
        let f2 = numeric_function::NumericFunction2D::new(Vec::new());
        functions_2d.insert("f2".to_string(), f2);

        let mut functions_3d = HashMap::new();
        let f3 = numeric_function::NumericFunction3D::new(Vec::new());
        functions_3d.insert("f3".to_string(), f3);

        let mut functions = HashMap::new();
        let f4 = numeric_function::NumericFunction::new(HashMap::new(), 0);
        functions.insert("f4".to_string(), f4);

        function_registry::FunctionRegistry {
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        }
    }

    #[test]
    fn parse_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = [")", "(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "+", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_not_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "not", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::Not(_)));
        if let Condition::Not(c) = expression {
            assert!(matches!(
                *c,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
        };
        assert_eq!(rest, &tokens[7..]);
    }

    #[test]
    fn parse_not_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "not", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "not", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "not", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_and_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::And(_, _)));
        if let Condition::And(x, y) = expression {
            assert!(matches!(
                *x,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
            assert!(matches!(
                *y,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(1)))
            ));
        };
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_and_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "and", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "and", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_or_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(expression, Condition::Or(_, _)));
        if let Condition::Or(x, y) = expression {
            assert!(matches!(
                *x,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
            ));
            assert!(matches!(
                *y,
                Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(1)))
            ));
        };
        assert_eq!(rest, &tokens[11..]);
    }

    #[test]
    fn parse_or_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "or", "(", "is_empty", "s0", ")", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = [
            "(", "or", "(", "is_empty", "s0", ")", "(", "is_empty", "s1", ")", "(", "is_empty",
            "s2", ")", ")", ")",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_eq_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Eq,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_eq_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ne_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "!=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Ne,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ne_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "!=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "!=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_gt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", ">", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Gt,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_gt_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_ge_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", ">=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Ge,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_ge_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", ">=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", ">=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_lt_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "<", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Lt,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_lt_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "<", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_le_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "<=", "2", "n0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Comparison(
                ComparisonOperator::Le,
                NumericExpression::Constant(2),
                NumericExpression::Variable(0)
            )
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_le_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "<=", "2", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "<=", "2", "n0", "n1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_in_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_in", "2", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsIn(
                ElementExpression::Constant(2),
                SetExpression::SetVariable(0)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_in_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_in", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_in", "0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_subset_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsSubset(
                SetExpression::SetVariable(0),
                SetExpression::SetVariable(1)
            ))
        ));
        assert_eq!(rest, &tokens[5..]);
    }

    #[test]
    fn parse_is_subset_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_subset", "e0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "e1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_subset", "s0", "s1", "s2", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }

    #[test]
    fn parse_is_empty_ok() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();
        let tokens: Vec<String> = ["(", "is_empty", "s0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_ok());
        let (expression, rest) = result.unwrap();
        assert!(matches!(
            expression,
            Condition::Set(SetCondition::IsEmpty(SetExpression::SetVariable(0)))
        ));
        assert_eq!(rest, &tokens[4..]);
    }

    #[test]
    fn parse_is_empty_err() {
        let metadata = generate_metadata();
        let registry = generate_registry();
        let parameters = generate_parameters();

        let tokens: Vec<String> = ["(", "is_empty", "e0", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());

        let tokens: Vec<String> = ["(", "is_empty", "s0", "s1", ")", ")"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        let result = parse_expression(&tokens, &metadata, &registry, &parameters);
        assert!(result.is_err());
    }
}
