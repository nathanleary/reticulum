package layers

import (
	"fmt"
	"math/rand"

	"github.com/nathanleary/reticulum/volume"
)

// NewDropoutLayer creates a new dropout layer.
func NewDropoutLayer(def LayerDef) Layer {
	if def.Type != Dropout {
		panic(fmt.Errorf("Invalid layer type: %s != input", def.Type))
	}

	// Cast layer config
	conf, ok := def.LayerConfig.(*DropoutLayerConfig)
	if !ok {
		panic(fmt.Errorf("Invalid layer config: expected DropoutLayerConfig got %T", conf))
	}

	n := def.Output.Size()
	return &dropoutLayer{conf, def.Input, def.Output, make([]bool, n, n), nil, nil}
}

// DropoutLayerConfig contains the dropout probablity.
type DropoutLayerConfig struct {
	DropoutProbability float64
}

type dropoutLayer struct {
	config *DropoutLayerConfig

	input   volume.Dimensions
	output  volume.Dimensions
	dropped []bool

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (l *dropoutLayer) Type() LayerType {
	return Dropout
}

func (l *dropoutLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	vol2 := vol.Clone()
	n := vol.Size()

	if training {
		// Perform dropout based on probabilty
		for i := 0; i < n; i++ {
			if rand.Float64() < l.config.DropoutProbability {
				vol2.SetByIndex(i, 0.0)
				l.dropped[i] = true
			} else {
				l.dropped[i] = false
			}
		}
	} else {
		// scale the activations during prediction
		for i := 0; i < n; i++ {
			vol2.MultByIndex(i, l.config.DropoutProbability)
		}
	}

	l.outVol = vol2
	return l.outVol
}

func (l *dropoutLayer) Backward() {

	// Need to set the gradients to zero
	l.inVol.ZeroGrad()
	chainGrad := l.outVol
	n := l.inVol.Size()

	// Apply dropouts to input volume
	for i := 0; i < n; i++ {
		if !l.dropped[i] {

			// copy over the gradient
			l.inVol.SetGradByIndex(i, chainGrad.GetGradByIndex(i))
		}
	}
}

func (l *dropoutLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
