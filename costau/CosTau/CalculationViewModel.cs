using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Windows.ApplicationModel.Resources;
using Windows.UI.Xaml;

namespace CosTau
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

        // for `EvalUnlessCalledAgainBeforeStarted`
        private CancellationTokenSource _cancellationTokenSource;

        public CalculationViewModel()
        {
            this.Update();
        }

        private async void Update()
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

            var evalResult = await EvalUnlessCalledAgainBeforeStarted(exprTrim);
            if (evalResult == null)
            {
                // we have a new expression that we're trying to evaluate
                return;
            }

            if (evalResult.HasFailed)
            {
                this.ResultText = ResourceLoader.GetForCurrentView().GetString("CalculationFailedText");
                this.ResultVisibility = Visibility.Visible;
                return;
            }

            var resultParts = new List<string>();
            if (evalResult.SimplifiedExpression != null)
            {
                resultParts.Add($"= {evalResult.SimplifiedExpression}");
            }
            if (evalResult.Approximation != null)
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

        private async Task<Ffi.EvalResult> EvalUnlessCalledAgainBeforeStarted(string expression)
        {
            if (_cancellationTokenSource != null)
            {
                // Cancel previous eval because we do not care about it anymore
                // now that we have a new expression to evaluate.
                _cancellationTokenSource.Cancel();
            }

            _cancellationTokenSource = new CancellationTokenSource();

            try
            {
                var myToken = _cancellationTokenSource.Token;
                return await Task.Run(() =>
                {
                    myToken.ThrowIfCancellationRequested();
                    return Ffi.Eval(expression);
                });
            }
            catch (OperationCanceledException)
            {
                // a newer expression is being evaluated
                return null;
            }
            catch (Exception)
            {
                // display failure
                return new Ffi.EvalResult(null, null, true);
            }
        }

        private void OnPropertyChanged(string name)
        {
            this.PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}
