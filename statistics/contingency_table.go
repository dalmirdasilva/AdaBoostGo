package statistics

import "fmt"

/**
 * matrix[0] = Condition Negative
 * matrix[1] = Condition Positive
 *
 * matrix[?][0] = Predicted Condition Negative
 * matrix[?][1] = Predicted Condition Positive
 *
 * matrix[0][0] = True Negative
 * matrix[1][1] = True Positive
 *
 * matrix[0][1] = False Negative
 * matrix[1][0] = False Positive
 *
 * @constructor
 */
type ContingencyTable struct {
    table [2][2]uint
}

func NewContingencyTable() ContingencyTable {
    return ContingencyTable{}
}

func (c *ContingencyTable) TruePositive() uint {
    return c.table[1][1]
}

func (c *ContingencyTable) FalsePositive() uint {
    return c.table[0][1]
}

func (c *ContingencyTable) TrueNegative() uint {
    return c.table[0][0]
}

func (c *ContingencyTable) FalseNegative() uint {
    return c.table[1][0]
}

func (c *ContingencyTable) AddPrediction(y, h int) {
    c.table[c.classToIndex(y)][c.classToIndex(h)]++
}

func (c *ContingencyTable) OutcomePositive() uint {
    return c.TruePositive() + c.FalsePositive()
}

func (c *ContingencyTable) OutcomeNegative() uint {
    return c.TrueNegative() + c.FalseNegative()
}

func (c *ContingencyTable) TotalPopulation() uint {
    return c.table[0][0] + c.table[0][1] + c.table[1][0] + c.table[1][1]
}

func (c *ContingencyTable) PredictedConditionPositive() uint {
    return c.TruePositive() + c.FalsePositive()
}

func (c *ContingencyTable) PredictedConditionNegative() uint {
    return c.FalseNegative() + c.TrueNegative()
}

func (c *ContingencyTable) ConditionPositive() uint {
    return c.TruePositive() + c.FalseNegative()
}

func (c *ContingencyTable) ConditionNegative() uint {
    return c.FalsePositive() + c.TrueNegative()
}

/**
 * Prevalence = E Condition positive / E Total population.
 *
 * @returns {number}
 */
func (c *ContingencyTable) Prevalence() float64 {
    return float64(c.ConditionPositive()) / float64(c.TotalPopulation())
}

/**
 * True positive rate (TPR), Sensitivity, Recall =  E True positive / E Condition positive.
 *
 * @returns {number}
 */
func (c *ContingencyTable) TruePositiveRate() float64 {
    return float64(c.TruePositive()) / float64(c.ConditionPositive())
}

func (c *ContingencyTable) Recall() float64 {
    return c.TruePositiveRate()
}

func (c *ContingencyTable) Sensitivity() float64 {
    return c.TruePositiveRate()
}

/**
 * False positive rate (FPR), Fall-out = E False positive / E Condition negative.
 *
 * @returns {number}
 */
func (c *ContingencyTable) FalsePositiveRate() float64 {
    return float64(c.FalsePositive()) / float64(c.ConditionNegative())
}

func (c *ContingencyTable) FallOut() float64 {
    return c.FalsePositiveRate()
}

/**
 * False negative rate (FNR), Miss rate = E False negative / E Condition positive.
 *
 * @returns {number}
 */
func (c *ContingencyTable) FalseNegativeRate() float64 {
    return float64(c.FalseNegative()) / float64(c.ConditionPositive())
}

/**
 * True negative rate (TNR), Specificity (SPC) = E True negative / E Condition negative.
 *
 * @returns {number}
 */
func (c *ContingencyTable) TrueNegativeRate() float64 {
    return float64(c.TrueNegative()) / float64(c.ConditionNegative())
}

func (c *ContingencyTable) Specificity() float64 {
    return c.TrueNegativeRate()
}

/**
 * Accuracy (ACC) = E True positive + E True negative / E Total population.
 *
 * @returns {number}
 */
func (c *ContingencyTable) Accuracy() float64 {
    return float64(c.TruePositive() + c.TrueNegative()) / float64(c.TotalPopulation())
}

/**
 * Positive predictive value (PPV), Precision = E True positive / E Test outcome positive.
 *
 * @returns {number}
 */
func (c *ContingencyTable) PositivePredictiveValue() float64 {
    return float64(c.TruePositive()) / float64(c.OutcomePositive())
}

func (c *ContingencyTable) Precision() float64 {
    return c.PositivePredictiveValue()
}

/**
 * False discovery rate (FDR) = E False positive / E Test outcome positive.
 *
 * @returns {number}
 */
func (c *ContingencyTable) FalseDiscoveryRate() float64 {
    return float64(c.FalsePositive()) / float64(c.OutcomePositive())
}

/**
 * False omission rate (FOR) = E False negative / E Test outcome negative.
 *
 * @returns {number}
 */
func (c *ContingencyTable) FalseOmissionRate() float64 {
    return float64(c.FalseNegative()) / float64(c.OutcomeNegative())
}

/**
 * Negative predictive value (NPV) = E True negative / E Test outcome negative.
 *
 * * @returns {number}
 */
func (c *ContingencyTable) NegativePredictiveValue() float64 {
    return float64(c.TrueNegative()) / float64(c.OutcomeNegative())
}

/**
 * Positive likelihood ratio (LR+) = TPR / FPR.
 *
 * @returns {number}
 */
func (c *ContingencyTable) PositiveLikelihoodRatio() float64 {
    return c.TruePositiveRate() / c.FalsePositiveRate()
}

/**
 * Negative likelihood ratio (LR−) = FNR / TNR.
 *
 * @returns {number}
 */
func (c *ContingencyTable) NegativeLikelihoodRatio() float64 {
    return c.FalseNegativeRate() / c.TrueNegativeRate()
}

/**
 * Diagnostic odds ratio (DOR) = LR+ / LR−.
 *
 * @returns {number}
 */
func (c *ContingencyTable) DiagnosticOddsRatio() float64 {
    return c.PositiveLikelihoodRatio() / c.NegativeLikelihoodRatio()
}

/**
 * To string.
 *
 * @returns {string}
 */
func (c *ContingencyTable) String() string {
    return fmt.Sprintf("\nTotal population: %d\t" +
    "\nCondition positive: %d\t" +
    "\nCondition negative: %d\t" +
    "\nPredicted Condition positive: %d\t" +
    "\nPredicted Condition negative: %d\t" +
    "\nTrue positive: %d\t" +
    "\nTrue negative: %d\t" +
    "\nFalse Negative: %d\t" +
    "\nFalse Positive: %d\t" +
    "\nPrevalence = Σ Condition positive / Σ Total population: %f\t" +
    "\nTrue positive rate (TPR) = Σ True positive / Σ Condition positive: %f\t" +
    "\nFalse positive rate (FPR) = Σ False positive / Σ Condition negative: %f\t" +
    "\nFalse negative rate (FNR) = Σ False negative / Σ Condition positive: %f\t" +
    "\nTrue negative rate (TNR) = Σ True negative / Σ Condition negative: %f\t" +
    "\nAccuracy (ACC) = Σ True positive + Σ True negative / Σ Total population: %f\t" +
    "\nPositive predictive value (PPV) = Σ True positive / Σ Test outcome positive: %f\t" +
    "\nFalse discovery rate (FDR) = Σ False positive / Σ Test outcome positive: %f\t" +
    "\nFalse omission rate (FOR) = Σ False negative / Σ Test outcome negative: %f\t" +
    "\nNegative predictive value (NPV) = Σ True negative / Σ Test outcome negative: %f\t" +
    "\nPositive likelihood ratio (LR+) = TPR / FPR: %f\t" +
    "\nNegative likelihood ratio (LR−) = FNR / TNR: %f\t" +
    "\nDiagnostic odds ratio (DOR) = LR+ / LR−: %f\t",
        c.TotalPopulation(),
        c.ConditionPositive(),
        c.ConditionNegative(),
        c.PredictedConditionPositive(),
        c.PredictedConditionNegative(),
        c.TruePositive(),
        c.TrueNegative(),
        c.FalseNegative(),
        c.FalsePositive(),
        c.Prevalence(),
        c.TruePositiveRate(),
        c.FalsePositiveRate(),
        c.FalseNegativeRate(),
        c.TrueNegativeRate(),
        c.Accuracy(),
        c.PositivePredictiveValue(),
        c.FalseDiscoveryRate(),
        c.FalseOmissionRate(),
        c.NegativePredictiveValue(),
        c.PositiveLikelihoodRatio(),
        c.NegativeLikelihoodRatio(),
        c.DiagnosticOddsRatio())
}

/**
 * Classes are 1 or -1. To use the class as the index of que binary confusion matrix we need to convert them
 * into 0 or 1.
 *
 * @param k
 * @returns {number}
 */
func (c *ContingencyTable) classToIndex(k int) uint {
    if k > 0 {
        return 1
    }
    return 0
}