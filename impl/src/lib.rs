extern crate either;
extern crate num_bigint;
extern crate num_integer;
extern crate num_rational;
extern crate num_traits;

pub mod lexer;
pub mod node;
pub mod parser;

mod ratio2flt;
mod result;

pub use result::EvalResult;

use std::{thread, time};
use std::os::raw::c_char;
use std::ffi::CStr;

#[repr(C)]
pub struct FloatResult {
    val: f64,
    success: bool,
}

#[no_mangle]
pub unsafe extern fn calculator_eval(expr: *const c_char) -> FloatResult {
    // to test if this is running on the UI thread
    thread::sleep(time::Duration::from_secs(1));

    let expr = match CStr::from_ptr(expr).to_str() {
        Ok(val) => val,
        Err(_) => return FloatResult { val: 0f64, success: false },
    };

    let lexer = lexer::Lexer::new(expr);

    let mut tokens = Vec::<lexer::Token>::new();
    for result in lexer {
        let token = match result {
            Ok(val) => val,
            Err(_) => return FloatResult { val: 0f64, success: false },
        };

        tokens.push(token);
    }

    let parser = parser::Parser::new(&tokens);
    let root_node = match parser.parse() {
        Ok(val) => val,
        Err(_) => return FloatResult { val: 0f64, success: false },
    };

    let eval = root_node/* .deep_reduce() */.eval();

    FloatResult { val: eval.val, success: true }
}
