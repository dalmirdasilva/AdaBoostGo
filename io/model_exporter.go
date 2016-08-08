package io

import (
    "github.com/dalmirdasilva/AdaBoostGo/static/model"
    "github.com/dalmirdasilva/AdaBoostGo/classifier"
    "github.com/golang/protobuf/proto"
    "encoding/json"
    "io/ioutil"
    "log"
)

type ModelExporter struct {
}

func NewModelExporter() ModelExporter {
    return ModelExporter{}
}

func (e *ModelExporter) populateProto(fileName string, classifier classifier.AdaBoost, numberOfFeatures uint) *dom_distiller.AdaBoostProto {
    adaBoostProto := dom_distiller.AdaBoostProto{}
    adaBoostProto.NumFeatures = new(int32)
    *adaBoostProto.NumFeatures = int32(numberOfFeatures)
    adaBoostProto.NumStumps = new(int32)
    *adaBoostProto.NumStumps = int32(len(classifier.WeakClassifiers))
    for _, weakClassifier := range classifier.WeakClassifiers {
        stumpProto := dom_distiller.StumpProto{}
        stumpProto.FeatureNumber = new(int32)
        *stumpProto.FeatureNumber = int32(weakClassifier.GetFeatureNumber())
        stumpProto.Split = new(float64)
        *stumpProto.Split = weakClassifier.GetSplit()
        stumpProto.Weight = new(float64)
        *stumpProto.Weight = weakClassifier.GetAlpha()
        adaBoostProto.Stump = append(adaBoostProto.Stump, &stumpProto)
    }
    return &adaBoostProto
}

func (e *ModelExporter) ExportToProto(fileName string, classifier classifier.AdaBoost, numberOfFeatures uint) {
    adaBoostProto := e.populateProto(fileName, classifier, numberOfFeatures)
    buf, err := proto.Marshal(adaBoostProto)
    if err != nil {
        log.Fatal(err)
    }
    ioutil.WriteFile(fileName, buf, 0600)
}

func (e *ModelExporter) ExportToJSON(fileName string, classifier classifier.AdaBoost, numberOfFeatures uint) {
    var model = make(map[string]interface{})
    model["num_features"] = numberOfFeatures
    model["num_stumps"] = len(classifier.WeakClassifiers)
    stumps := []map[string]interface{}{}
    for _, weakClassifier := range classifier.WeakClassifiers {
        var stump = make(map[string]interface{})
        stump["feature_number"] = weakClassifier.GetFeatureNumber()
        stump["split"] = weakClassifier.GetSplit()
        stump["weight"] = weakClassifier.GetAlpha()
        stumps = append(stumps, stump)
    }
    model["stump"] = stumps
    buf, err := json.Marshal(model)
    if err != nil {
        log.Fatal(err)
    }
    ioutil.WriteFile(fileName, buf, 0600)
}

