extern crate costau_core;
extern crate num_traits;

use costau_core::lexer::Lexer;
use costau_core::parser::Parser;
use costau_core::node::Node;
use num_traits::One;
use std::env;

fn main() {
    let expr = env::args().skip(1).collect::<Vec<_>>().join(" ");
    println!("{}", expr);

    let lexer = Lexer::new(&expr);
    let tokens = lexer.map(|r| r.unwrap()).collect::<Vec<_>>();

    let parser = Parser::new(&tokens);
    let root_node = parser.parse().unwrap();

    let simplified = match root_node.simplify() {
        Ok(x) => x,
        Err(err) => {
            println!("= (error: {:?})", err);
            return;
        }
    };
    if simplified.did_something {
        println!("= {}", simplified.result);
    }

    let needs_approx = if let Node::Num { val, .. } = &simplified.result {
        // if integer, already simplified to the maximum
        !val.denom().is_one() && *val.denom() != (-1).into()
    } else {
        true
    };
    if needs_approx {
        match simplified.result.eval() {
            Ok(eval) => println!("≈ {}", eval),
            Err(err) => println!("≈ (error: {:?})", err),
        }
    }
}
