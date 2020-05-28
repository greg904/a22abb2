use a22abb2_core::lexer::{Lexer, Token};
use a22abb2_core::parser::Parser;
use std::ffi::CStr;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

pub struct EvalSuccess {
    simplified_expr: String,
    approx: String,
}

type EvalResult = Result<EvalSuccess, ()>;

#[no_mangle]
pub unsafe extern "C" fn a22abb2_evalresult_free(r: *mut EvalResult) {
    // let the compiler drop the box
    let _ = Box::from_raw(r);
}

#[no_mangle]
pub unsafe extern "C" fn a22abb2_evalresult_has_failed(r: *mut EvalResult) -> bool {
    (*r).is_err()
}

#[cfg(windows)]
unsafe fn alloc_csharp_str(s: &str) -> *mut c_char {
    let char_count = s.len();

    // create string for consumption on C# side
    let chars_with_nul = char_count + 1;
    let bytes = chars_with_nul * mem::size_of::<c_char>();
    let out = slice::from_raw_parts_mut(winapi::um::combaseapi::CoTaskMemAlloc(bytes) as *mut c_char, chars_with_nul);

    // fill string
    let mut in_bytes = s.bytes();
    for b in out.iter_mut() {
        *b = in_bytes.next().map(|x| x as c_char).unwrap_or(0); // NUL terminator
    }

    out.as_mut_ptr()
}

#[cfg(not(windows))]
unsafe fn alloc_csharp_str(s: &str) -> *mut c_char {
    panic!("unsupported platform");
}

#[no_mangle]
pub unsafe extern "C" fn a22abb2_evalresult_get_approx(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(success) => alloc_csharp_str(&success.approx),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn a22abb2_evalresult_get_simplified_expr(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(success) => alloc_csharp_str(&success.simplified_expr),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn a22abb2_eval(expr: *const c_char) -> *mut EvalResult {
    let expr = CStr::from_ptr(expr);
    let expr = expr.to_str().unwrap();

    let lexer = Lexer::new(expr);
    let mut tokens: Vec<Token> = Vec::new();
    for result in lexer {
        let token = match result {
            Ok(x) => x,
            Err(_) => return Box::into_raw(Box::new(Err(()))),
        };
        tokens.push(token);
    }

    let parser = Parser::new(&tokens);
    let root_node = match parser.parse() {
        Ok(x) => x,
        Err(_) => return Box::into_raw(Box::new(Err(()))),
    };
    let simplified = match root_node.simplify() {
        Ok(x) => x,
        Err(_) => return Box::into_raw(Box::new(Err(()))),
    };
    let approx = match simplified.eval() {
        Ok(x) => x.val,
        Err(_) => return Box::into_raw(Box::new(Err(()))),
    };

    let r = EvalSuccess {
        simplified_expr: simplified.to_string(),
        approx: approx.to_string(),
    };
    Box::into_raw(Box::new(Ok(r)))
}
