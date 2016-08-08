package evaluation

import (
    "github.com/dalmirdasilva/AdaBoostGo/classifier"
    "github.com/dalmirdasilva/AdaBoostGo/config"
    "github.com/dalmirdasilva/AdaBoostGo/statistics"
    "github.com/dalmirdasilva/AdaBoostGo/utils"
    "math"
)

type Evaluator struct {
    classifier       *classifier.AdaBoost
    contingencyTable statistics.ContingencyTable
    threshold        float64
}

func NewEvaluator(classifier *classifier.AdaBoost) Evaluator {
    return Evaluator{classifier: classifier, threshold: math.MaxFloat64}
}

// Calculates the confusion matrix for a classifier and a test set.
func (e *Evaluator) Evaluate(testSet [][]float64) statistics.ContingencyTable {
    e.contingencyTable = statistics.NewContingencyTable()
    for _, sample := range testSet {
        y := int(sample[len(sample) - 1])
        var h int
        if config.USE_THRESHOLD_CLASSIFICATION {
            h = e.classifyUsingThreshold(sample)
        } else {
            h = e.classifyNormally(sample)
        }
        e.contingencyTable.AddPrediction(y, h)
    }
    return e.contingencyTable
};

// Computes the threshold for a classifier.
func (e *Evaluator) getThreshold() float64 {
    if e.threshold == math.MaxFloat64 {
        e.threshold = 0
        for _, weakClassifier := range e.classifier.WeakClassifiers {
            e.threshold += weakClassifier.GetAlpha() / 2
        }
    }
    return e.threshold
}

// Classify a sample using threshold.
func (e *Evaluator) classifyUsingThreshold(sample []float64) int {
    score := 0.0
    for _, weakClassifier := range e.classifier.WeakClassifiers {
        if sample[weakClassifier.GetFeatureNumber()] > weakClassifier.GetSplit() {
            score += weakClassifier.GetAlpha()
        }
    }
    if score > e.getThreshold() {
        return 1
    }
    return -1
}

// Classify a sample just using the sign of the sum od weights.
func (e *Evaluator) classifyNormally(sample []float64) int {
    if e.classifier.Classify(sample) > 0 {
        return 1
    }
    return -1
}

// Gets the list of used features by the classifier.
func (e *Evaluator) GetUsedFeatureNumbers(unique bool) []uint {
    var usedFeatureNumbers []uint
    for _, weakClassifier := range e.classifier.WeakClassifiers {
        usedFeatureNumbers = append(usedFeatureNumbers, weakClassifier.GetFeatureNumber())
    }
    if unique {
        return utils.RemoveDuplicates(usedFeatureNumbers)
    }
    return usedFeatureNumbers
}

// Gets the map of feature number occurrences of the classifier.
func (e *Evaluator) GetFeatureOccurrences() map[uint]uint {
    occurrences := make(map[uint]uint)
    featureNumbers := e.GetUsedFeatureNumbers(false)
    for _, featureNumber := range featureNumbers {
        occurrences[featureNumber]++
    }
    return occurrences
}
