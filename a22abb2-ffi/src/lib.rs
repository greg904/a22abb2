extern crate a22abb2;

use std::f64;
use std::ffi::CStr;
use std::os::raw::c_char;

use a22abb2::lexer::{Lexer, Token};
use a22abb2::parser::Parser;

#[no_mangle]
pub unsafe extern fn eval(expr: *const c_char) -> f64 {
    let expr = CStr::from_ptr(expr);
    let expr = expr.to_str().unwrap();

    let lexer = Lexer::new(expr);
    let mut tokens: Vec<Token> = Vec::new();
    for result in lexer {
        let token = match result {
            Ok(val) => val,
            Err(_) => return f64::NAN,
        };
        tokens.push(token);
    }
    
    let parser = Parser::new(&tokens);
    let root_node = match parser.parse() {
        Ok(val) => val,
        Err(_) => return f64::NAN,
    };
    root_node.eval().val
}
