extern crate costau_core;

use std::env;

use costau_core::lexer::Lexer;
use costau_core::parser::Parser;

fn main() {
    let expr = env::args().skip(1).collect::<Vec<_>>().join(" ");
    println!("Original expression: {}", expr);

    let lexer = Lexer::new(&expr);
    let tokens = lexer.map(|r| r.unwrap()).collect::<Vec<_>>();

    let parser = Parser::new(&tokens);
    let root_node = parser.parse().unwrap();

    let simplified = root_node.simplify().unwrap();
    if simplified.did_something {
        println!("Simplified expression: {}", simplified.result);
    }

    match simplified.result.eval() {
        Ok(eval) => println!("Expression result: {}", eval),
        Err(_) => println!("Expression result: (error)"),
    }
}
