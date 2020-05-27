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

                this.UpdateResult();
            }
        }

        private Visibility historyVisibility = Visibility.Collapsed;
        public Visibility HistoryVisibility
        {
            get => this.historyVisibility;
            private set
            {
                if (this.historyVisibility == value)
                {
                    return;
                }
                this.historyVisibility = value;
                this.OnPropertyChanged(nameof(HistoryVisibility));
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

        private bool hasFailed;
        public bool HasFailed
        {
            get => this.hasFailed;
            private set
            {
                if (this.hasFailed == value)
                {
                    return;
                }
                this.hasFailed = value;
                this.OnPropertyChanged(nameof(HasFailed));
            }
        }

        private void UpdateResult()
        {
            var exprTrim = this.expression.Trim();
            if (exprTrim == "")
            {
                this.HistoryVisibility = Visibility.Collapsed;
                return;
            }

            // TODO: do not call this on the UI thread
            var evalResult = Ffi.Eval(exprTrim);

            this.HasFailed = evalResult.HasFailed;
            if (this.HasFailed)
            {
                this.ResultText = ResourceLoader.GetForCurrentView().GetString("CalculationFailedText");
                this.HistoryVisibility = Visibility.Visible;
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
            var approximationNormalized = ExpressionNormalizeRegex.Replace(evalResult.Approximation.ToString(), "");
            if (!expressionNormalized.Equals(approximationNormalized) && !simplificationNormalized.Equals(approximationNormalized))
            {
                resultParts.Add($"≈ {evalResult.Approximation}");
            }

            if (resultParts.Count > 0)
            {
                this.HistoryVisibility = Visibility.Visible;
                this.ResultText = string.Join("\n", resultParts);
            }
            else
            {
                this.HistoryVisibility = Visibility.Collapsed;
            }
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
