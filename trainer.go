package reticulum

import (
	"math"
	"time"

	"github.com/nathanleary/reticulum/layers"
	"github.com/nathanleary/reticulum/volume"
)

type Trainer interface {
	Train(vol *volume.Volume, lossFn LossFunc) TrainingResults
}

func NewTrainer(net Network, opts ...OptionFunc) Trainer {
	if net == nil {
		panic("network cannot be nil")
	}

	// Read opts
	baseOpts := &Options{Method: SGD, LearningRate: 0.01, BatchSize: 1, Momentum: 0.9, Ro: 0.95, Eps: 1e-8, Beta1: 0.9, Beta2: 0.999}
	for _, optFn := range opts {
		optFn(baseOpts)
	}

	var isRegression bool
	l := net.Layers()
	if _, ok := l[net.Size()-1].(layers.RegressionLossLayer); ok {
		isRegression = true
	}
	return &trainer{net, baseOpts, 0, [][]float64{}, [][]float64{}, isRegression}
}

type trainer struct {
	net  Network
	opts *Options

	// iteration counter
	k int

	// last iteration gradients (used for momentum calculations)
	gsum [][]float64

	// used in adam or adadelta
	xsum [][]float64

	// check if regression is used
	regression bool
}

type LossFunc func(net Network) float64

func LabeledLossFunc(label int) LossFunc {
	return func(net Network) float64 {
		return net.Backward(label)
	}
}

func RegressionLossFunc(y []float64) LossFunc {
	return func(net Network) float64 {
		return net.MultiDimensionalLoss(y)
	}
}

func (t *trainer) Train(vol *volume.Volume, lossFunc LossFunc) TrainingResults {
	start := time.Now()
	t.net.Forward(vol, true)
	fwdTime := time.Now().Sub(start)

	start = time.Now()
	costLoss := lossFunc(t.net)
	bwdTime := time.Now().Sub(start)

	t.k++
	var l1DecayLoss, l2DecayLoss float64
	if t.k%t.opts.BatchSize == 0 {
		pgList := t.net.GetResponse()

		// initialize lists for accumulators. Will only be done once on first iteration
		if len(t.gsum) == 0 && t.opts.Method == SGD || t.opts.Momentum > 0.0 {
			for i := 0; i < len(pgList); i++ {
				t.gsum = append(t.gsum, make([]float64, len(pgList[i].Weights)))
				if t.opts.Method == Adam || t.opts.Method == Adadelta {
					t.xsum = append(t.xsum, make([]float64, len(pgList[i].Weights)))
				} else {
					t.xsum = append(t.xsum, []float64{})
				}
			}
		}

		// perform an update for all sets of weights
		for i, pg := range pgList {
			p := pg.Weights
			g := pg.Gradients

			// learning rate for some parameters.
			l1DecayMul, l2DecayMul := pg.L1DecayMul, pg.L2DecayMul
			l1Decay := t.opts.L1Decay * l1DecayMul
			l2Decay := t.opts.L2Decay * l2DecayMul

			for j := 0; j < len(p); j++ {
				// accumulate weight decay loss
				l2DecayLoss += l2Decay * p[j] * p[j] / 2.0
				l1DecayLoss += l1Decay * math.Abs(p[j])
				l1Grad, l2Grad := l1Decay, l2Decay*p[j]
				if p[j] <= 0 {
					l1Grad *= -1
				}

				// raw batch gradient
				gij := (l2Grad + l1Grad + g[j]) / float64(t.opts.BatchSize)

				meth := t.opts.Method
				gsumi, xsumi := t.gsum[i], t.xsum[i]
				if meth == Adam {

					// update biased first moment estimate
					gsumi[j] = gsumi[j]*t.opts.Beta1 + (1-t.opts.Beta1)*gij

					// update biased second moment estimate
					xsumi[j] = xsumi[j]*t.opts.Beta2 + (1-t.opts.Beta2)*gij*gij

					// correct bias first moment estimate
					biasCorr1 := gsumi[j] * (1 - math.Pow(t.opts.Beta1, float64(t.k)))

					// correct bias second moment estimate
					biasCorr2 := xsumi[j] * (1 - math.Pow(t.opts.Beta2, float64(t.k)))

					dx := -t.opts.LearningRate * biasCorr1 / (math.Sqrt(biasCorr2) + t.opts.Eps)
					p[j] += dx
				} else if meth == Adagrad {
					// update biased first moment estimate
					gsumi[j] = gsumi[j] + gij*gij

					dx := -t.opts.LearningRate / (math.Sqrt(gsumi[j]) + t.opts.Eps) * gij
					p[j] += dx
				} else if meth == Windowgrad {
					// this is adagrad but with a moving window weighted average
					// so the gradient is not accumulated over the entire history of the run.
					// it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
					gsumi[j] = t.opts.Ro*gsumi[j] + (1-t.opts.Ro)*gij*gij

					// eps added for better conditioning
					dx := -t.opts.LearningRate / math.Sqrt(gsumi[j]+t.opts.Eps) * gij
					p[j] += dx
				} else if meth == Adadelta {
					gsumi[j] = t.opts.Ro*gsumi[j] + (1-t.opts.Ro)*gij*gij
					dx := -math.Sqrt((xsumi[j]+t.opts.Eps)/(gsumi[j]+t.opts.Eps)) * gij
					xsumi[j] = t.opts.Ro*xsumi[j] + (1-t.opts.Ro)*dx*dx // yes, xsum lags behind gsum by 1.
					p[j] += dx
				} else if meth == Netsterov {
					dx := gsumi[j]
					gsumi[j] = gsumi[j]*t.opts.Momentum + t.opts.LearningRate*gij
					dx = t.opts.Momentum*dx - (1.0+t.opts.Momentum)*gsumi[j]
					p[j] += dx
				} else {

					// Assume SGD
					if t.opts.Momentum > 0.0 {
						// momentum update

						// step
						dx := t.opts.Momentum*gsumi[j] - t.opts.LearningRate*gij

						// back this up for next iteration of momentum
						gsumi[j] = dx

						// apply corrected gradient
						p[j] += dx
					} else {
						// vanilla sgd
						p[j] += -t.opts.LearningRate * gij
					}
				}

				// zero out gradient so that we can begin accumulating anew
				g[j] = 0.0
			}
		}
	}
	return TrainingResults{
		ForwardTime:  fwdTime,
		BackwardTime: bwdTime,
		L1DecayLoss:  l1DecayLoss,
		L2DecayLoss:  l2DecayLoss,
		CostLost:     costLoss,
		TotalLoss:    costLoss + l1DecayLoss + l2DecayLoss,
	}
}

type TrainingResults struct {
	ForwardTime  time.Duration
	BackwardTime time.Duration
	L1DecayLoss  float64
	L2DecayLoss  float64
	CostLost     float64
	TotalLoss    float64
}
