using System.Runtime.InteropServices;

namespace A22abb2
{
    class A22abb2
    {
        [DllImport("a22abb2_ffi.dll", EntryPoint = "eval", CharSet = CharSet.Ansi)]
        public static extern double Eval(string expression);
    }
}
