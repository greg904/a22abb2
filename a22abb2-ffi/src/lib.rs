use a22abb2_core::lexer::{Lexer, Token};
use a22abb2_core::parser::Parser;
use std::f64;
use std::ffi::CStr;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

pub struct EvalSuccess {
    expr_simplified: String,
    approx: f64,
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
pub unsafe extern fn a22abb2_evalresult_get_approx(r: *mut EvalResult) -> f64 {
    match &*r {
        Ok(success) => success.approx,
        Err(_) => f64::NAN,
    }
}

#[cfg(windows)]
unsafe fn alloc_csharp_str(byte_count: usize) -> *mut c_char {
    winapi::um::combaseapi::CoTaskMemAlloc(byte_count) as *mut c_char
}

#[cfg(not(windows))]
unsafe fn alloc_csharp_str(_byte_count: usize) -> *mut c_char {
    panic!("Unsupported platform");
}

#[no_mangle]
pub unsafe extern fn a22abb2_evalresult_get_expr_simplified(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(success) => {
            let char_count = success.expr_simplified.len();

            // create string for consumption on C# side
            let chars_with_nul = char_count + 1;
            let bytes = chars_with_nul * mem::size_of::<c_char>();
            let out = slice::from_raw_parts_mut(alloc_csharp_str(bytes), chars_with_nul);
            
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
    let simplified = root_node.simplify();
    let simplified_str = simplified.to_string();
    let approx = simplified.eval().val;

    let r = EvalSuccess {
        expr_simplified: simplified_str,
        approx,
    };
    Box::into_raw(Box::new(Ok(r)))
}
