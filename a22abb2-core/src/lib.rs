extern crate either;
extern crate float_cmp;
extern crate num_bigint;
extern crate num_integer;
extern crate num_rational;
extern crate num_traits;

pub mod lexer;
pub mod node;
pub mod parser;
mod ratio2flt;

#[cfg(test)]
mod tests {
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    /// A regression test.
    #[test]
    fn run_all_test_calculations() {
        let tests_str = include_str!("../../test-calculations.txt");
        for test in tests_str.lines() {
            // test lexer
            let mut tokens = Vec::new();
            for r in Lexer::new(test) {
                match r {
                    Ok(token) => tokens.push(token),
                    Err(err) => panic!("failed to lex expression {}: {:?}", test, err),
                }
            }
            // test parser
            let node = match Parser::new(&tokens).parse() {
                Ok(x) => x,
                Err(err) => panic!("failed to parse expression {}: {:?}", test, err),
            };
            // first round of checking if it can be evaluated
            match node.eval() {
                Ok(ground_truth) => {
                    if !ground_truth.val.is_finite() {
                        panic!("approx before simplify is infinite for {}", test);
                    }
                    // make sure that it can be simplified
                    match node.clone().simplify() {
                        Ok(simplified) => {
                            // second round of checking if it can be evaluated
                            match simplified.eval() {
                                Ok(simplified_result) => {
                                    // BTW, let's check if we simplified correctly
                                    let mut is_equal = (simplified_result.val - ground_truth.val).abs() < 0.1;
                                    if !is_equal && ground_truth.val != 0.0 {
                                        let rel_error =
                                            ((simplified_result.val - ground_truth.val) / ground_truth.val).abs();
                                        is_equal = rel_error < 0.1;
                                    }
                                    if !is_equal {
                                        panic!("got incorrect simplification for {}: got {} which evaluates to {} instead of {}", test, simplified, simplified_result.val, ground_truth.val);
                                    }
                                }
                                Err(err) => panic!("failed to eval {} after simplify: {:?}", test, err),
                            }
                        },
                        Err(err) => panic!("failed to simplify {}: {:?}", test, err),
                    }
                }
                Err(err) => panic!("failed to eval {} before simplify: {:?}", test, err),
            }
        }
    }
}
