package main

import (
    "github.com/dalmirdasilva/GoAdaboostClassifier/classifier"
    "github.com/dalmirdasilva/GoAdaboostClassifier/evaluation"
    "github.com/dalmirdasilva/GoAdaboostClassifier/statistics"
    "github.com/dalmirdasilva/GoAdaboostClassifier/config"
    "github.com/dalmirdasilva/GoAdaboostClassifier/utils"
    "github.com/dalmirdasilva/GoAdaboostClassifier/io"
    "math/rand"
    "time"
    "fmt"
    "log"
    "os"
)

func main() {

    rand.Seed(time.Now().UTC().UnixNano())
    trainingDataFilePath := os.Args[1]

    samples := utils.ReadSamples(trainingDataFilePath)

    for _, sample := range samples {
        if sample[len(sample) - 1] > 5 {
            sample[len(sample) - 1] = 1
        } else {
            sample[len(sample) - 1] = -1
        }
    }

    numberOfSamples := len(samples)
    if numberOfSamples < 1 {
        log.Fatal("At least one sample is needed.")
    }

    numberOfFeatures := len(samples[0]) - 1
    if numberOfFeatures < 1 {
        log.Fatal("At least feature is needed.")
    }

    testSize := int(config.TEST_PERCENT * float64(len(samples)))
    testSamples := samples[:testSize]

    trainingSamples := samples[testSize:]
    fmt.Println(len(trainingSamples))

    adaBoost := classifier.NewAdaBoost(config.NUM_OF_WEAK_CLASSIFIERS)
    adaBoost.Train(trainingSamples)

    evaluator := evaluation.NewEvaluator(&adaBoost)
    contingencyTable := evaluator.Evaluate(testSamples)

    fmt.Println(evaluator.GetUsedFeatureNumbers(false))
    fmt.Println(evaluator.GetFeatureOccurrences())
    fmt.Println(contingencyTable.String())

    analyzer := statistics.NewFeaturesAnalyzer()
    stats, distribution := analyzer.Analyze(trainingSamples)
    fmt.Println(stats, distribution)

    //for _, sample := range currentRound {
    //    sample = append(sample, adaBoost.Classify(sample))
    //    fmt.Println(sample)
    //}
    //
    ////
    for _, weakClassifier := range adaBoost.WeakClassifiers {
        fmt.Println(weakClassifier.String())
    }

    //var variableRelations []statistics.VariableRelations
    //for i := 0; i < numberOfFeatures; i++ {
    //    for j := i + 1; j < numberOfFeatures; j++ {
    //        variableRelations = append(variableRelations, analyzer.Correlation(uint(i), uint(j), trainingSamples, stats))
    //    }
    //}
    //sort.Sort(statistics.SortableVariableRelations(variableRelations))
    //for _, variableRelation := range variableRelations {
    //    if variableRelation.Cov != 0 {
    //        b, _ := json.Marshal(variableRelation)
    //        fmt.Println(string(b) + ", ")
    //    }
    //}
    //
    //for i := 0; i < numberOfFeatures; i++ {
    //    variableRelation := analyzer.Correlation(uint(i), uint(29), trainingSamples, stats)
    //    b, _ := json.Marshal(variableRelation)
    //    fmt.Println(string(b) + ", ")
    //}
    //
    me := io.NewModelExporter()
    //me.ExportToProto("/tmp/proto.bin", adaBoost, uint(numberOfFeatures))
    me.ExportToJSON("/tmp/proto.json", adaBoost, uint(numberOfFeatures))
}