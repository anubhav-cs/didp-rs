use didp_parser::variable;
use didp_search::astar;
use std::env;
use std::fs;
use std::process;

fn main() {
    let mut args = env::args();
    args.next();
    let domain = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a domain file name.");
        process::exit(1);
    });
    let problem = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a problem file name.");
        process::exit(1);
    });
    let config = args.next().unwrap_or_else(|| {
        eprintln!("Didn't get a config file name.");
        process::exit(1);
    });
    let domain = fs::read_to_string(domain).unwrap_or_else(|e| {
        eprintln!("Couldn't read a domain file: {:?}", e);
        process::exit(1);
    });
    let domain = yaml_rust::YamlLoader::load_from_str(&domain).unwrap_or_else(|e| {
        eprintln!("Coundn't read a domain file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(domain.len(), 1);
    let domain = &domain[0];
    let problem = fs::read_to_string(problem).unwrap_or_else(|e| {
        eprintln!("Could'nt read a problem file: {:?}", e);
        process::exit(1);
    });
    let problem = yaml_rust::YamlLoader::load_from_str(&problem).unwrap_or_else(|e| {
        eprintln!("Coundn't read a problem file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(problem.len(), 1);
    let problem = &problem[0];
    let config = fs::read_to_string(config).unwrap_or_else(|e| {
        eprintln!("Could'nt read a config file: {:?}", e);
        process::exit(1);
    });
    let config = yaml_rust::YamlLoader::load_from_str(&config).unwrap_or_else(|e| {
        eprintln!("Coundn't read a config file: {:?}", e);
        process::exit(1);
    });
    assert_eq!(config.len(), 1);
    let config = &config[0];
    let cost_type = didp_parser::CostType::load_from_yaml(domain).unwrap_or_else(|e| {
        eprintln!("Coundn't load a cost type: {:?}", e);
        process::exit(1);
    });
    match cost_type {
        didp_parser::CostType::Integer => {
            let model = didp_parser::Model::<variable::Integer>::load_from_yaml(domain, problem)
                .unwrap_or_else(|e| {
                    eprintln!("Coundn't load a model: {:?}", e);
                    process::exit(1);
                });
            let solution = astar::astar(&model, config).unwrap_or_else(|e| {
                eprintln!("Error: {:?}", e);
                process::exit(1);
            });
            match solution {
                Some((cost, transitions)) => {
                    println!("cost: {}", cost);
                    for transition in transitions {
                        println!("{}", transition.name);
                    }
                }
                None => {
                    println!("no solution");
                }
            }
        }
        didp_parser::CostType::Continuous => {
            let model =
                didp_parser::Model::<variable::OrderedContinuous>::load_from_yaml(domain, problem)
                    .unwrap_or_else(|e| {
                        eprintln!("Coundn't load a model: {:?}", e);
                        process::exit(1);
                    });
            let solution = astar::astar(&model, config).unwrap_or_else(|e| {
                eprintln!("Error: {:?}", e);
                process::exit(1);
            });
            match solution {
                Some((cost, transitions)) => {
                    println!("cost: {}", cost);
                    for transition in transitions {
                        println!("{}", transition.name);
                    }
                }
                None => {
                    println!("no solution");
                }
            }
        }
    }
}
