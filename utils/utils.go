package utils

import (
    "os"
    "encoding/csv"
    "bufio"
    "log"
    "strconv"
    "io"
    "fmt"
    "math/rand"
)

func RemoveDuplicates(elements []uint) []uint {
    encountered := map[uint]bool{}
    result := []uint{}
    for v := range elements {
        ok, _ := encountered[elements[v]]
        if !ok {
            encountered[elements[v]] = true
            result = append(result, elements[v])
        }
    }
    return result
}

func ReadSamples(fileName string) (d [][]float64) {
    f, e := os.Open(fileName)
    if e != nil {
        log.Fatal(e)
    }
    r := csv.NewReader(bufio.NewReader(f))
    for {
        record, err := r.Read()
        if err == io.EOF {
            break
        }
        var entry []float64
        for value := range record {
            f, err := strconv.ParseFloat(record[value], 64)
            if err == nil {
                entry = append(entry, f)
            } else {
                fmt.Println(err)
            }
        }
        d = append(d, entry)
    }
    return
}

func ShuffleSamples(samples [][]float64) {
    for i := range samples {
        j := rand.Intn(i + 1)
        samples[i], samples[j] = samples[j], samples[i]
    }
}