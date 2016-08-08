package statistics

type VariableRelations struct {
    X uint
    Y uint
    Cov float64
    Cor float64
}

type SortableVariableRelations []VariableRelations

func (s SortableVariableRelations) Len() int {
    return len(s)
}

func (s SortableVariableRelations) Less(i, j int) bool {
    return s[i].Cor < s[j].Cor
}

func (s SortableVariableRelations) Swap(i, j int) {
    s[i], s[j] = s[j], s[i]
}

