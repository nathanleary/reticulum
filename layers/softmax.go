package layers

import (
	"fmt"
	"math"

	"github.com/nathanleary/reticulum/volume"
)

// NewSoftmaxLayer creates a new softmax layer.
// This is a classifier, with N discrete classes from 0 to N-1. It gets a stream
// of N incoming numbers and computes the softmax function (exponentiate and
// normalize to sum to 1 as probabilities should)
func NewSoftmaxLayer(def LayerDef) Layer {
	if def.Type != SoftMax {
		panic(fmt.Errorf("invalid layer type: %s != softmax", def.Type))
	} else if def.LayerConfig == nil {
		panic(fmt.Errorf("invalid layer config"))
	}

	// Get config
	conf, ok := def.LayerConfig.(*softMaxLayerConfig)
	if !ok {
		panic("invalid LayerConfig for softMaxLayerConfig")
	}

	n := def.Input.Size()
	return &softmaxLayer{
		conf:   conf,
		inDim:  def.Input,
		outDim: volume.Dimensions{X: 1, Y: 1, Z: n},
		inVol:  nil,
		outVol: nil,
		es:     []float64{},
	}
}

// NewSoftmaxLayerConfig creates a new LayerConfig config with the given options.
func NewSoftmaxLayerConfig(classes int, opts ...LayerOptionFunc) LayerConfig {
	if classes <= 0 {
		panic("class count must be greater than 0")
	}

	conf := &softMaxLayerConfig{
		Classes: classes,
	}
	for i := 0; i < len(opts); i++ {
		err := opts[i](conf)
		if err != nil {
			panic(err)
		}
	}
	return conf
}

// softMaxLayerConfig stores the config info for softmax layers
type softMaxLayerConfig struct {
	Classes int
}

// GetSoftMaxPrediction returns the argmax prediction for the softmax layer.
func GetSoftMaxPrediction(layer Layer) int {
	// this is a convenience function for returning the argmax
	// prediction, assuming the last layer of the net is a softmax
	softmax, ok := layer.(*softmaxLayer)
	if !ok {
		panic("expected Softmax layer")
	}

	p := softmax.outVol.Weights()
	maxv, maxi := p[0], 0
	for index := 0; index < len(p); index++ {
		if p[index] > maxv {
			maxv = p[index]
			maxi = index
		}
	}
	return maxi
}

type softmaxLayer struct {
	conf   *softMaxLayerConfig
	inDim  volume.Dimensions
	outDim volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	es []float64
}

func (l *softmaxLayer) Type() LayerType {
	return SoftMax
}

func (l *softmaxLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol

	n := l.outDim.Z
	volA := volume.NewVolume(l.outDim, volume.WithZeros())

	// compute max activation
	as := vol.Weights()
	aMax := as[0]
	for i := 0; i < n; i++ {
		if as[i] > aMax {
			aMax = as[i]
		}
	}

	// compute exponentials (carefully to not blow up)
	es := make([]float64, n, n)
	esum := 0.0
	for i := 0; i < n; i++ {
		e := math.Exp(as[i] - aMax)
		esum += e
		es[i] = e
	}

	// normalize and output to sum to one
	for i := 0; i < n; i++ {
		es[i] /= esum
		volA.SetByIndex(i, es[i])
	}

	// save these for backprop
	l.es = es
	l.outVol = volA
	return l.outVol
}

func (l *softmaxLayer) Loss(index int) float64 {
	if index < 0 || index >= l.outDim.Size() {
		panic(fmt.Errorf("Invalid dimension index: %d", index))
	}

	// compute and accumulate gradient wrt weights and bias of this layer
	// zero out the gradient of input Vol
	l.inVol.ZeroGrad()

	n := l.outDim.Z
	for i := 0; i < n; i++ {
		indicator := 0.0
		if i == index {
			indicator = 1.0
		}

		l.inVol.SetGradByIndex(i, -(indicator - l.es[i]))
	}

	// loss is the class negative log likelihood
	return -math.Log(l.es[index])
}

func (l *softmaxLayer) Backward() {
	panic(fmt.Errorf("Unsupported operation"))
}

func (l *softmaxLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
