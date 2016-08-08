package classifier

import (
    "fmt"
    "math"
)

type WeakClassifier struct {
    featureNumber uint
    split         float64
    error         float64
    alpha         float64
}

func NewWeakClassifier(featureNumber uint, split float64) WeakClassifier {
    return WeakClassifier{featureNumber: featureNumber, split: split}
}

// Set weight α_{t} based on the error
// Computes the following equation:
// \alpha_{t} = \frac{1}{2}\ln\left( \frac{1 - \epsilon_{t}(h_{t})}{\epsilon_{t}(h_{t})}\right)
func (c *WeakClassifier) ComputeAlpha() {
    c.alpha = 0.5 * math.Log((1.0 - c.error) / c.error)
}

func (c *WeakClassifier) Classify(sample []float64) int {
    if sample[c.featureNumber] > c.split {
        return 1
    }
    return -1
}

func (c *WeakClassifier) ClassifyWithAlpha(sample []float64) float64 {
    return float64(c.Classify(sample)) * c.alpha
}

func (c *WeakClassifier) IncreaseError(amount float64) {
    c.error += amount
}

func (c *WeakClassifier) SetFeatureNumber(featureNumber uint) {
    c.featureNumber = featureNumber
}

func (c *WeakClassifier) GetFeatureNumber() uint {
    return c.featureNumber
}

func (c *WeakClassifier) SetSplit(split float64) {
    c.split = split
}

func (c *WeakClassifier) GetSplit() float64 {
    return c.split
}

func (c *WeakClassifier) SetError(error float64) {
    c.error = error
}

func (c *WeakClassifier) GetError() float64 {
    return c.error
}

func (c *WeakClassifier) SetAlpha(alpha float64) {
    c.alpha = alpha
}

func (c *WeakClassifier) GetAlpha() float64 {
    return c.alpha
}

func (c *WeakClassifier) String() string {
    return fmt.Sprintf("featureNumber: %d, split: %f, error: %f, apha: %f", c.featureNumber, c.split, c.error, c.alpha)
}
