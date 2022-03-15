use crate::numeric_function;
use crate::state;
use crate::variable;
use crate::yaml_util;
use std::collections;
use std::fmt;
use std::str;
use yaml_rust::Yaml;

pub struct FunctionRegistry<T: variable::Numeric> {
    pub functions_1d: collections::HashMap<String, numeric_function::NumericFunction1D<T>>,
    pub functions_2d: collections::HashMap<String, numeric_function::NumericFunction2D<T>>,
    pub functions_3d: collections::HashMap<String, numeric_function::NumericFunction3D<T>>,
    pub functions: collections::HashMap<String, numeric_function::NumericFunction<T>>,
}

impl<T: variable::Numeric> FunctionRegistry<T> {
    pub fn load_from_yaml(
        domain: &Yaml,
        problem: &Yaml,
        metadata: &state::StateMetadata,
    ) -> Result<FunctionRegistry<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut name_to_arg_types = collections::HashMap::new();
        let mut name_to_default_value = collections::HashMap::new();
        let domain = yaml_util::get_map(domain)?;
        for (key, value) in domain {
            let name = yaml_util::get_string(key)?;
            let value = yaml_util::get_map(value)?;
            let args = yaml_util::get_string_array_by_key(value, "args")?;
            if args.is_empty() {
                return Err(yaml_util::YamlContentErr::new(
                    "function has no arguments".to_string(),
                ));
            }
            let mut arg_types = Vec::with_capacity(args.len());
            for object in &args {
                if let Some(value) = metadata.name_to_element_variable.get(object) {
                    arg_types.push(*value);
                } else {
                    return Err(yaml_util::YamlContentErr::new(format!(
                        "object `{}` does not exist",
                        object
                    )));
                }
            }
            name_to_arg_types.insert(name.clone(), arg_types);
            if let Ok(value) = yaml_util::get_numeric_by_key(value, "default") {
                name_to_default_value.insert(name.clone(), value);
            } else {
                name_to_default_value.insert(name.clone(), T::zero());
            }
        }
        let mut functions_1d = collections::HashMap::new();
        let mut functions_2d = collections::HashMap::new();
        let mut functions_3d = collections::HashMap::new();
        let mut functions = collections::HashMap::new();
        let problem = yaml_util::get_map(problem)?;
        for (name, args_types) in name_to_arg_types {
            let value = yaml_util::get_yaml_by_key(problem, &name)?;
            let default = *name_to_default_value.get(&name).unwrap();
            if args_types.len() == 1 {
                let size = metadata.object_numbers[args_types[0]];
                let f = Self::load_function_1d_from_yaml(value, size, default)?;
                functions_1d.insert(name, f);
            } else if args_types.len() == 2 {
                let size_x = metadata.object_numbers[args_types[0]];
                let size_y = metadata.object_numbers[args_types[1]];
                let f = Self::load_function_2d_from_yaml(value, size_x, size_y, default)?;
                functions_2d.insert(name, f);
            } else if args_types.len() == 3 {
                let size_x = metadata.object_numbers[args_types[0]];
                let size_y = metadata.object_numbers[args_types[1]];
                let size_z = metadata.object_numbers[args_types[2]];
                let f = Self::load_function_3d_from_yaml(value, size_x, size_y, size_z, default)?;
                functions_3d.insert(name, f);
            } else {
                let size: Vec<usize> = args_types
                    .iter()
                    .map(|i| metadata.object_numbers[*i])
                    .collect();
                let f = Self::load_function_from_yaml(value, size, default)?;
                functions.insert(name, f);
            }
        }
        Ok(FunctionRegistry {
            functions_1d,
            functions_2d,
            functions_3d,
            functions,
        })
    }

    fn load_function_1d_from_yaml(
        value: &Yaml,
        size: usize,
        default: T,
    ) -> Result<numeric_function::NumericFunction1D<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut body: Vec<T> = (0..size).map(|_| default).collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize(args)?;
            let value = yaml_util::get_numeric(value)?;
            if args >= size {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{}` is greater than the number of the object for function",
                    args,
                )));
            }
            body[args] = value;
        }
        Ok(numeric_function::NumericFunction1D::new(body))
    }

    fn load_function_2d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        default: T,
    ) -> Result<numeric_function::NumericFunction2D<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut body: Vec<Vec<T>> = (0..size_x)
            .map(|_| (0..size_y).map(|_| default).collect())
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let value = yaml_util::get_numeric(value)?;
            if x >= size_x || y >= size_y {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {})` is greater than the numbers of objects for function",
                    x, y,
                )));
            }
            body[x][y] = value;
        }
        Ok(numeric_function::NumericFunction2D::new(body))
    }

    fn load_function_3d_from_yaml(
        value: &Yaml,
        size_x: usize,
        size_y: usize,
        size_z: usize,
        default: T,
    ) -> Result<numeric_function::NumericFunction3D<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let mut body: Vec<Vec<Vec<T>>> = (0..size_x)
            .map(|_| {
                (0..size_y)
                    .map(|_| (0..size_z).map(|_| default).collect())
                    .collect()
            })
            .collect();
        let map = yaml_util::get_map(value)?;
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            let x = args[0];
            let y = args[1];
            let z = args[2];
            let value = yaml_util::get_numeric(value)?;
            if x >= size_x || y >= size_y || z >= size_z {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`({}, {}, {})` is greater than the numbers of objects for function",
                    x, y, z,
                )));
            }
            body[x][y][z] = value;
        }
        Ok(numeric_function::NumericFunction3D::new(body))
    }

    fn load_function_from_yaml(
        value: &Yaml,
        size: Vec<usize>,
        default: T,
    ) -> Result<numeric_function::NumericFunction<T>, yaml_util::YamlContentErr>
    where
        <T as str::FromStr>::Err: fmt::Debug,
    {
        let map = yaml_util::get_map(value)?;
        let mut body = collections::HashMap::with_capacity(map.len());
        for (args, value) in map {
            let args = yaml_util::get_usize_array(args)?;
            if args.len() != size.len() {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "expected `{}` arguments for function, but passed `{}`",
                    size.len(),
                    args.len()
                )));
            }
            let value = yaml_util::get_numeric(value)?;
            if args.iter().zip(size.iter()).any(|(a, b)| a >= b) {
                return Err(yaml_util::YamlContentErr::new(format!(
                    "`{:?}` is greater than the numbers of objects for function",
                    args,
                )));
            }
            body.insert(args, value);
        }
        Ok(numeric_function::NumericFunction::new(body, default))
    }
}