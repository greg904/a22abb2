using System;
using System.Runtime.InteropServices;

namespace A22abb2
{
    class Ffi
    {
        [DllImport("a22abb2_ffi.dll", EntryPoint = "a22abb2_evalresult_free")]
        private static extern void Native_EvalResult_Free(IntPtr ptr);

        [DllImport("a22abb2_ffi.dll", EntryPoint = "a22abb2_evalresult_has_failed")]
        private static extern bool Native_EvalResult_HasFailed(IntPtr ptr);

        [DllImport("a22abb2_ffi.dll", EntryPoint = "a22abb2_evalresult_get_result_val")]
        private static extern double Native_EvalResult_GetResultVal(IntPtr ptr);

        [DllImport("a22abb2_ffi.dll", EntryPoint = "a22abb2_evalresult_get_expr_simplified", CharSet = CharSet.Ansi)]
        private static extern string Native_EvalResult_GetExprSimplified(IntPtr ptr);

        [DllImport("a22abb2_ffi.dll", EntryPoint = "a22abb2_eval", CharSet = CharSet.Ansi)]
        private static extern IntPtr Native_Eval(string expression);

        public struct EvalResult
        {
            public double ResultValue;
            public string SimplifiedExpression;

            public bool HasFailed
            {
                get => this.SimplifiedExpression == null;
            }

            public EvalResult(double resultValue, string simplifiedExpression)
            {
                this.ResultValue = resultValue;
                this.SimplifiedExpression = simplifiedExpression;
            }
        }

        public static EvalResult Eval(string expression)
        {
            var ptr = Native_Eval(expression);
            try
            {
                if (Native_EvalResult_HasFailed(ptr))
                {
                    // error
                    return new EvalResult();
                }
                else
                {
                    // success
                    var resultVal = Native_EvalResult_GetResultVal(ptr);
                    var simplified = Native_EvalResult_GetExprSimplified(ptr);
                    return new EvalResult(resultVal, simplified);
                }
            }
            finally
            {
                Native_EvalResult_Free(ptr);
            }
        }
    }
}
