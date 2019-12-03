using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Reluclac
{
    class A22abb2
    {
        [DllImport("a22abb2_ffi.dll", EntryPoint = "eval", CharSet = CharSet.Ansi)]
        public static extern double Eval(string expression);
    }
}
