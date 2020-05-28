using System.Collections.Generic;
using System.ComponentModel;
using System.Text.RegularExpressions;
using Windows.ApplicationModel.Resources;
using Windows.UI.Xaml;

namespace A22abb2
{
    public sealed class CalculationViewModel : INotifyPropertyChanged
    {
        // Remove whitespace and parentheses, which can change when the node is
        // parsed and dumped back.
        // Note that this is not a very good way to check if two expressions
        // are different.
        // In the long term, we should really make it so that the Rust core
        // returns whether it did anything to the node to simplify it and
        // whether the evaluation folded nodes into numbers instead of
        // guessing with this heuristic.
        private static readonly Regex ExpressionNormalizeRegex = new Regex(@"[\s()]");

        public event PropertyChangedEventHandler PropertyChanged;

        private string expression = "";
        public string Expression
        {
            get => this.expression;
            set
            {
                if (this.expression == value)
                {
                    return;
                }
                this.expression = value;
                this.OnPropertyChanged(nameof(Expression));

                this.Update();
            }
        }

        private string expressionText;
        public string ExpressionText
        {
            get => this.expressionText;
            private set
            {
                if (this.expressionText == value)
                {
                    return;
                }
                this.expressionText = value;
                this.OnPropertyChanged(nameof(ExpressionText));
            }
        }

        private Visibility resultVisibility;
        public Visibility ResultVisibility
        {
            get => this.resultVisibility;
            private set
            {
                if (this.resultVisibility == value)
                {
                    return;
                }
                this.resultVisibility = value;
                this.OnPropertyChanged(nameof(ResultVisibility));
            }
        }

        private string resultText;
        public string ResultText
        {
            get => this.resultText;
            private set
            {
                if (this.resultText == value)
                {
                    return;
                }
                this.resultText = value;
                this.OnPropertyChanged(nameof(ResultText));
            }
        }

        private bool isEmpty;
        public bool IsEmpty
        {
            get => this.isEmpty;
            private set
            {
                if (this.isEmpty == value)
                {
                    return;
                }
                this.isEmpty = value;
                this.OnPropertyChanged(nameof(IsEmpty));
            }
        }

        public CalculationViewModel()
        {
            this.Update();
        }

        private void Update()
        {
            var exprTrim = this.expression.Trim();
            if (exprTrim == "")
            {
                this.ExpressionText = ResourceLoader.GetForCurrentView().GetString("ExpressionEmptyText");
                this.ResultVisibility = Visibility.Collapsed;
                this.IsEmpty = true;
                return;
            }
            else
            {
                this.ExpressionText = exprTrim;
                this.IsEmpty = false;
            }

            // TODO: do not call this on the UI thread
            var evalResult = Ffi.Eval(exprTrim);
            if (evalResult.HasFailed)
            {
                this.ResultText = ResourceLoader.GetForCurrentView().GetString("CalculationFailedText");
                this.ResultVisibility = Visibility.Visible;
                return;
            }

            // Only show the parts that are useful, which are the parts that
            // are not the same as the expression that the user typed.
            var expressionNormalized = ExpressionNormalizeRegex.Replace(this.expression, "");
            var resultParts = new List<string>();
            var simplificationNormalized = ExpressionNormalizeRegex.Replace(evalResult.SimplifiedExpression, "");
            if (!expressionNormalized.Equals(simplificationNormalized))
            {
                resultParts.Add($"= {evalResult.SimplifiedExpression}");
            }
            var approximationNormalized = ExpressionNormalizeRegex.Replace(evalResult.Approximation, "");
            if (!expressionNormalized.Equals(approximationNormalized) && !simplificationNormalized.Equals(approximationNormalized))
            {
                resultParts.Add($"≈ {evalResult.Approximation}");
            }

            if (resultParts.Count > 0)
            {
                this.ResultText = string.Join("\n", resultParts);
                this.ResultVisibility = Visibility.Visible;
            }
            else
            {
                this.ResultVisibility = Visibility.Collapsed;
            }
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
