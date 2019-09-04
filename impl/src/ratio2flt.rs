use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero, One};

use std::{f64, u64};

// TODO: add tests
pub fn ratio_to_f64(num: &BigRational) -> f64 {
    let is_negative = num.is_negative();
    let denom = num.denom().abs();
    let mut remaining = num.numer().abs();
    let mut fraction: u64 = 0;

    // The unsigned exponent. The real exponent is this value minus 1023.
    let mut exp: u16 = 1023 + 52;

    let mut remaining_shift_right: u32 = 0;

    'outer: while !remaining.is_zero() {
        while remaining < denom {
            // The mantissa on a float number is 52 bits.
            // If we don't have enough space to store the next binary digit,
            // then stop here because it is better to lose precision in the
            // least significant digits than the most significant digits.
            // Also, the exponent cannot be smaller than -1022.
            // See https://en.wikipedia.org/wiki/Double-precision_floating-point_format.
            if fraction & (1 << 53) != 0 || exp <= 2 {
                break 'outer;
            }

            // keep the same number, but add some place for a new binary digit

            if remaining_shift_right > 0 {
                remaining_shift_right -= 1;
            } else {
                remaining <<= 1;
            }

            fraction <<= 1;
            exp -= 1;
        }

        while remaining_shift_right > 0 {
            remaining >>= 1;
            remaining_shift_right -= 1;
        }

        let (mut div, rem) = remaining.div_rem(&denom);

        // If we lost less significants digits in the mantissa to add more
        // space for more significant digits, this will tell us whether we
        // must round up.
        let mut most_significant_lost_digit_is_one: Option<bool> = Option::None;

        loop {
            let div_u64 = div.to_u64();
            let add_result = div_u64.and_then(|n| fraction.checked_add(n));

            match add_result {
                Some(val) => fraction = val,
                None => {
                    // The exponent cannot be larger than 1023.
                    if exp >= 1023 + 1023 {
                        return f64::MAX;
                    }

                    let will_lose_one = (fraction as u8) & 1u8 == 1u8 ||
                        (&div & BigInt::one()).to_u8().unwrap() == 1u8;

                    if most_significant_lost_digit_is_one.is_none() {
                        most_significant_lost_digit_is_one = Some(will_lose_one);
                    }

                    // Don't shift it immediately because we might shift it to
                    // left later and we don't want to lose bits.
                    remaining_shift_right += 1;

                    div >>= 1;
                    fraction >>= 1;
                    exp += 1;

                    // try again with less precision
                    continue;
                }
            }

            if most_significant_lost_digit_is_one.unwrap_or(false) && fraction != u64::MAX {
                // round up (for example, 0.1111111... turns into 1, but not 0.1101111)
                fraction += 1;
            }

            break;
        }

        remaining = rem;
    }

    if fraction != 0 {
        // same as above
        let mut most_significant_lost_digit_is_one: Option<bool> = Option::None;

        // Make sure that the fraction can be represented with 53 bits or less
        // (it is 53 instead of 52 because only the mantissa is stored on the
        // floating point number so the last bit is implied to be a one).
        while fraction >= (1 << 53) {
            if exp >= 1023 + 1023 {
                return if !is_negative { f64::MAX } else { -f64::MIN };
            }

            let will_lose_one = (fraction as u8) & 1u8 == 1u8;

            if most_significant_lost_digit_is_one.is_none() {
                most_significant_lost_digit_is_one = Some(will_lose_one);
            }

            fraction >>= 1;
            exp += 1;
        }

        if most_significant_lost_digit_is_one.unwrap_or(false) && fraction < (1 << 53) - 1 {
            // round up
            fraction += 1;
        }

        // Make sure that the 53th bit on the fraction is a one, because
        // only the mantissa is stored on the floating point number.
        while fraction & (1 << 52) == 0 {
            if exp <= 2 {
                return if !is_negative { f64::MIN_POSITIVE } else { -f64::MIN_POSITIVE };
            }

            fraction <<= 1;
            exp -= 1;
        }
    } else {
        return 0.0;
    }

    let result_u64 = (fraction & 0x000F_FFFF_FFFF_FFFF)
        | ((u64::from(exp & 0b0111_1111_1111)) << 52)
        | (u64::from(is_negative) << 63);
    f64::from_bits(result_u64)
}
