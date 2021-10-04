package layers

import (
	"fmt"

	"github.com/nathanleary/reticulum/volume"
)

// WithDecay sets the L1 & L2 decay for the fully conn or conv layer
func WithDecay(l1 float64, l2 float64) LayerOptionFunc {
	return func(lc LayerConfig) error {
		switch conf := lc.(type) {
		case *fullyConnLayerConfig:
			conf.L1DecayMult = l1
			conf.L2DecayMult = l2
		case *convLayerConfig:
			conf.L1DecayMult = l1
			conf.L2DecayMult = l2
		default:
			return fmt.Errorf("Invalid LayerConfig for FullyConnLayer")
		}
		return nil
	}
}

// WithBias sets the preferred bias for the layer
func WithBias(bias float64) LayerOptionFunc {
	return func(lc LayerConfig) error {
		switch conf := lc.(type) {
		case *fullyConnLayerConfig:
			conf.PreferredBias = bias
		case *convLayerConfig:
			conf.PreferredBias = bias
		default:
			return fmt.Errorf("Invalid LayerConfig for FullyConnLayer")
		}
		return nil
	}
}

// NewFullyConnectedLayerConfig creates a new LayerConfig config with the given options.
func NewFullyConnectedLayerConfig(neurons int, opts ...LayerOptionFunc) LayerConfig {
	if neurons <= 0 {
		panic("Neuron count must be greater than 0")
	}

	conf := &fullyConnLayerConfig{
		Neurons:       neurons,
		L1DecayMult:   0.0,
		L2DecayMult:   1.0,
		PreferredBias: 0.0,
	}
	for i := 0; i < len(opts); i++ {
		err := opts[i](conf)
		if err != nil {
			panic(err)
		}
	}
	return conf
}

// fullyConnLayerConfig stores the config info for fully connected layers
type fullyConnLayerConfig struct {
	Neurons       int
	L1DecayMult   float64
	L2DecayMult   float64
	PreferredBias float64
}

// NewFullyConnectedLayer creates a new fully connected layer.
func NewFullyConnectedLayer(def LayerDef) Layer {

	// Validate input
	if def.Type != FullyConnected {
		panic(fmt.Errorf("Invalid layer type: %s != fc", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for a fully connected layer"))
	} else if def.LayerConfig == nil {
		panic(fmt.Errorf("Config cannot be nil for a fully connected layer"))
	}

	// Get config
	conf, ok := def.LayerConfig.(*fullyConnLayerConfig)
	if !ok {
		panic("Invalid LayerConfig for fullyConnLayer")
	}

	// Output dimensions
	outDepth := conf.Neurons
	outDim := volume.Dimensions{X: 1, Y: 1, Z: outDepth}

	bias := conf.PreferredBias
	var filters []*volume.Volume
	for i := 0; i < outDepth; i++ {
		filters = append(filters, volume.NewVolume(volume.Dimensions{X: 1, Y: 1, Z: def.Input.Size()}))
	}

	biases := volume.NewVolume(volume.Dimensions{X: 1, Y: 1, Z: outDepth}, volume.WithInitialValue(bias))
	return &fullyConnLayer{conf, def.Input, outDim, nil, nil, filters, biases}
}

type fullyConnLayer struct {
	conf   *fullyConnLayerConfig
	input  volume.Dimensions
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	filters []*volume.Volume
	biases  *volume.Volume
}

func (*fullyConnLayer) Type() LayerType {
	return FullyConnected
}

func (l *fullyConnLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	A := volume.NewVolume(l.output, volume.WithZeros())

	w := vol.Weights()
	for i := 0; i < l.output.Size(); i++ {
		var a float64
		wi := l.filters[i].Weights()
		for d := 0; d < l.input.Size(); d++ {
			a += w[d] * wi[d]
		}
		a += l.biases.GetByIndex(i)
		A.SetByIndex(i, a)
	}

	l.outVol = A
	return l.outVol
}

func (l *fullyConnLayer) Backward() {
	l.inVol.ZeroGrad()

	numInputs := l.input.Size()
	for i := 0; i < l.output.Z; i++ {
		tfi := l.filters[i]
		chainGrad := l.outVol.GetGradByIndex(i)
		for d := 0; d < numInputs; d++ {
			l.inVol.AddGradByIndex(d, tfi.GetByIndex(d)*chainGrad)
			tfi.AddGradByIndex(d, l.inVol.GetByIndex(d)*chainGrad)
		}
		l.biases.AddGradByIndex(i, chainGrad)
	}
}

func (l *fullyConnLayer) GetResponse() []LayerResponse {
	var resp []LayerResponse
	for i := 0; i < l.output.Z; i++ {
		resp = append(resp, LayerResponse{
			Weights:    l.filters[i].Weights(),
			Gradients:  l.filters[i].Gradients(),
			L1DecayMul: l.conf.L1DecayMult,
			L2DecayMul: l.conf.L2DecayMult,
		})
	}
	resp = append(resp, LayerResponse{
		Weights:    l.biases.Weights(),
		Gradients:  l.biases.Gradients(),
		L1DecayMul: 0.0,
		L2DecayMul: 0.0,
	})
	return resp
}
