using System;
using System.Runtime.InteropServices;

namespace CosTau
{
    class Ffi
    {
        [DllImport("costau_ffi.dll", EntryPoint = "costau_evalresult_free", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Native_EvalResult_Free(IntPtr ptr);

        [DllImport("costau_ffi.dll", EntryPoint = "costau_evalresult_has_failed", CallingConvention = CallingConvention.Cdecl)]
        [return:MarshalAs(UnmanagedType.I1)]
        private static extern bool Native_EvalResult_HasFailed(IntPtr ptr);

        [DllImport("costau_ffi.dll", EntryPoint = "costau_evalresult_get_approx", CallingConvention = CallingConvention.Cdecl)]
        private static extern string Native_EvalResult_GetApprox(IntPtr ptr);

        [DllImport("costau_ffi.dll", EntryPoint = "costau_evalresult_get_simplified_expr", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern string Native_EvalResult_GetSimplifiedExpr(IntPtr ptr);

        [DllImport("costau_ffi.dll", EntryPoint = "costau_eval", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern IntPtr Native_Eval(string expression);

        public struct EvalResult
        {
            public string Approximation;
            public string SimplifiedExpression;

            public bool HasFailed
            {
                get => this.Approximation == null && this.SimplifiedExpression == null;
            }

            public EvalResult(string resultValue, string simplifiedExpression)
            {
                this.Approximation = resultValue;
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
                    var approx = Native_EvalResult_GetApprox(ptr);
                    var simplified = Native_EvalResult_GetSimplifiedExpr(ptr);
                    return new EvalResult(approx, simplified);
                }
            }
            finally
            {
                Native_EvalResult_Free(ptr);
            }
        }
    }
}
