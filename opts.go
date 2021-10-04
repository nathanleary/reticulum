package reticulum

type TrainingMethod string

// Available training methods
const (
	SGD        TrainingMethod = "sgd"
	Adam       TrainingMethod = "adam"
	Adagrad    TrainingMethod = "adagrad"
	Adadelta   TrainingMethod = "adadelta"
	Windowgrad TrainingMethod = "windowgrad"
	Netsterov  TrainingMethod = "netsterov"
)

type OptionFunc func(*Options)

type Options struct {
	Method       TrainingMethod
	LearningRate float64
	L1Decay      float64
	L2Decay      float64
	BatchSize    int

	Momentum float64
	Ro       float64
	Eps      float64
	Beta1    float64
	Beta2    float64
}

func WithMethod(m TrainingMethod) OptionFunc {
	return func(opts *Options) {
		opts.Method = m
	}
}

func WithLearningRate(rate float64) OptionFunc {
	return func(opts *Options) {
		opts.LearningRate = rate
	}
}

func WithDecay(l1, l2 float64) OptionFunc {
	return func(opts *Options) {
		opts.L1Decay = l1
		opts.L2Decay = l2
	}
}

func WithBatchSize(size int) OptionFunc {
	return func(opts *Options) {
		opts.BatchSize = size
	}
}

func WithMomentum(m float64) OptionFunc {
	return func(opts *Options) {
		opts.Momentum = m
	}
}

func WithEps(e float64) OptionFunc {
	return func(opts *Options) {
		opts.Eps = e
	}
}

func WithAdam(ro, beta1, beta2 float64) OptionFunc {
	return func(opts *Options) {
		opts.Method = Adam
		opts.Ro = ro
		opts.Beta1 = beta1
		opts.Beta2 = beta2
	}
}
