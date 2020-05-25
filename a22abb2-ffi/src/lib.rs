extern crate a22abb2_core;
extern crate winapi;

use std::f64;
use std::ffi::CStr;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use winapi::um::combaseapi::CoTaskMemAlloc;

use a22abb2_core::lexer::{Lexer, Token};
use a22abb2_core::parser::Parser;

pub struct EvalSuccess {
    expr_simplified: String,
    result_val: f64,
}

type EvalResult = Result<EvalSuccess, ()>;

#[no_mangle]
pub unsafe extern fn a22abb2_evalresult_free(r: *mut EvalResult) {
    // let the compiler drop the box
    let _ = Box::from_raw(r);
}

#[no_mangle]
pub unsafe extern fn a22abb2_evalresult_has_failed(r: *mut EvalResult) -> bool {
    match &*r {
        Ok(_) => false,
        Err(_) => true,
    }
}

#[no_mangle]
pub unsafe extern fn a22abb2_evalresult_get_result_val(r: *mut EvalResult) -> f64 {
    match &*r {
        Ok(success) => success.result_val,
        Err(_) => f64::NAN,
    }
}

#[no_mangle]
pub unsafe extern fn a22abb2_evalresult_get_expr_simplified(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(success) => {
            let char_count = success.expr_simplified.len();

            // create string for consumption on C# side
            let chars_with_nul = char_count + 1;
            let bytes = chars_with_nul * mem::size_of::<c_char>();
            let out = slice::from_raw_parts_mut(CoTaskMemAlloc(bytes) as *mut c_char, chars_with_nul);
            
            // fill string
            let mut in_bytes = success.expr_simplified.bytes();
            for b in out.iter_mut() {
                *b = in_bytes.next()
                    .map(|x| x as c_char)
                    .unwrap_or(0); // NUL terminator
            }

            out.as_mut_ptr()
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern fn a22abb2_eval(expr: *const c_char) -> *mut EvalResult {
    let expr = CStr::from_ptr(expr);
    let expr = expr.to_str().unwrap();
    
    let lexer = Lexer::new(expr);
    let mut tokens: Vec<Token> = Vec::new();
    for result in lexer {
        let token = match result {
            Ok(val) => val,
            Err(_) => return Box::into_raw(Box::new(Err(()))),
        };
        tokens.push(token);
    }
    
    let parser = Parser::new(&tokens);
    let root_node = match parser.parse() {
        Ok(val) => val,
        Err(_) => return Box::into_raw(Box::new(Err(()))),
    };
    let reduced = root_node.deep_reduce();
    let reduced_str = reduced.to_string();
    let result_val = reduced.eval().val;

    let r = EvalSuccess {
        expr_simplified: reduced_str,
        result_val,
    };
    Box::into_raw(Box::new(Ok(r)))
}
