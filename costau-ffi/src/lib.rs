use costau_core::lexer::{Lexer, Token};
use costau_core::parser::Parser;
use costau_core::node::Node;
use num_traits::One;
use std::ffi::CStr;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::slice;

pub struct EvalSuccess {
    simplified_expr: Option<String>,
    approx: Option<String>,
}

type EvalResult = Result<EvalSuccess, ()>;

#[no_mangle]
pub unsafe extern "C" fn costau_evalresult_free(r: *mut EvalResult) {
    // let the compiler drop the box
    let _ = Box::from_raw(r);
}

#[no_mangle]
pub unsafe extern "C" fn costau_evalresult_has_failed(r: *mut EvalResult) -> bool {
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
pub unsafe extern "C" fn costau_evalresult_get_approx(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(EvalSuccess { approx: Some(s), .. }) => alloc_csharp_str(&s),
        _ => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn costau_evalresult_get_simplified_expr(r: *mut EvalResult) -> *mut c_char {
    match &*r {
        Ok(EvalSuccess { simplified_expr: Some(s), .. }) => alloc_csharp_str(&s),
        _ => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn costau_eval(expr: *const c_char) -> *mut EvalResult {
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

    let (simplified_expr, did_simplify) = match root_node.simplify() {
        Ok(x) => (x.result, x.did_something),
        Err(_) => return Box::into_raw(Box::new(Err(()))),
    };

    let needs_approx = if let Node::Num { val, .. } = &simplified_expr {
        // if integer, already simplified to the maximum
        !val.denom().is_one() && *val.denom() != (-1).into()
    } else {
        true
    };
    let approx = if needs_approx {
        simplified_expr.eval().ok()
            .map(|x| x.val.to_string())
    } else {
        None
    };

    let r = EvalSuccess {
        simplified_expr: Some(simplified_expr)
            .filter(|_| did_simplify)
            .map(|x| x.to_string()),
        approx,
    };
    Box::into_raw(Box::new(Ok(r)))
}
