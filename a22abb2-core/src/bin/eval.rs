extern crate a22abb2_core;

use std::env;

use a22abb2_core::lexer::Lexer;
use a22abb2_core::parser::Parser;

fn main() {
    let expr = env::args().skip(1).collect::<Vec<_>>().join(" ");
    println!("Original expression: {}", expr);

    let lexer = Lexer::new(&expr);
    let tokens = lexer.map(|r| r.unwrap()).collect::<Vec<_>>();

    let parser = Parser::new(&tokens);
    let root_node = parser.parse().unwrap();

    let simplified = root_node.simplify().unwrap();
    println!("Simplified expression: {}", simplified);

    let eval = simplified.eval().unwrap();
    println!("Expression result: {}", eval);
}
