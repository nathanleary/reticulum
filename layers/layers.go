package layers

import (
	"github.com/nathanleary/reticulum/volume"
)

// LayerType describes the network layer
type LayerType string

// LayerType enums
const (
	FullyConnected    LayerType = "fc"
	LocalResponseNorm LayerType = "lrn"
	Dropout           LayerType = "dropout"
	Input             LayerType = "input"
	SoftMax           LayerType = "softmax"
	Regression        LayerType = "regression"
	Conv              LayerType = "conv"
	Pool              LayerType = "pool"
	ReLU              LayerType = "relu"
	Sigmoid           LayerType = "sigmoid"
	Tanh              LayerType = "tanh"
	Maxout            LayerType = "maxout"
	SVM               LayerType = "svm"
)

// LayerConfig stores layer specific config
type LayerConfig interface{}

// LayerOptionFunc provides for options in LayerConfig
type LayerOptionFunc func(LayerConfig) error

// LayerDef outlines the layer type, size and config.
type LayerDef struct {
	Type LayerType

	// Input dimensions
	Input volume.Dimensions

	// Output dim
	Output volume.Dimensions

	// Activation type
	Activation LayerType

	// Dropout adds a dropout layer afterwards with the given config
	Dropout *DropoutLayerConfig

	// Maxout adds a maxout layer afterwards with the given config
	Maxout *MaxoutLayerConfig

	// LayerConfig contains layer specific requirements
	LayerConfig LayerConfig
}

// Layer represents a layer in the neural network.
type Layer interface {
	Type() LayerType
	Forward(vol *volume.Volume, training bool) *volume.Volume
	Backward()
	GetResponse() []LayerResponse
}

// LossLayer extends the Layer interface with the Loss function
type LossLayer interface {
	Layer
	Loss(index int) float64
}

// RegressionLossLayer extends the Layer interface with the Loss function
type RegressionLossLayer interface {
	Layer
	MultiDimensionalLoss(losses []float64) float64
	DimensionalLoss(index int, value float64) float64
}

// LayerResponse represents the layer parameters (weights) and gradients.
type LayerResponse struct {
	Weights    []float64
	Gradients  []float64
	L1DecayMul float64
	L2DecayMul float64
}

// ActivateLayers adds activation, dropout layers, etc.
func ActivateLayers(defs []LayerDef) []LayerDef {
	var newDefs []LayerDef
	for _, def := range defs {

		// add an fc layer here, there is no reason the user should
		// have to worry about this and we almost always want to
		if def.Type == SoftMax || def.Type == SVM {
			switch conf := def.LayerConfig.(type) {
			case *softMaxLayerConfig:
				newDefs = append(newDefs, LayerDef{
					Type:        FullyConnected,
					LayerConfig: NewFullyConnectedLayerConfig(conf.Classes),
				})
			case *svmLayerConfig:
				newDefs = append(newDefs, LayerDef{
					Type:        FullyConnected,
					LayerConfig: NewFullyConnectedLayerConfig(conf.Classes),
				})
			default:
				panic("invalid LayerConfig")
			}
		}

		// add an fc layer here, there is no reason the user should
		// have to worry about this and we almost always want to
		if def.Type == Regression {
			conf, ok := def.LayerConfig.(*regressionLayerConfig)
			if !ok {
				panic("invalid LayerConfig for svmLayerConfig")
			}
			newDefs = append(newDefs, LayerDef{
				Type:        FullyConnected,
				LayerConfig: NewFullyConnectedLayerConfig(conf.Neurons),
			})
		}

		// Update bias
		if def.Type == FullyConnected || def.Type == Conv {
			// ReLUs like a bit of positive bias to get gradients early
			// otherwise it's technically possible that a relu unit will never turn on (by chance)
			// and will never get any gradient and never contribute any computation. Dead relu.
			if def.Activation == ReLU {
				switch conf := def.LayerConfig.(type) {
				case *fullyConnLayerConfig:
					conf.PreferredBias = 0.1
				case *convLayerConfig:
					conf.PreferredBias = 0.1
				default:
				}
			}
		}

		// Add def
		newDefs = append(newDefs, def)

		// Add activation layer
		if def.Activation != "" {
			switch def.Activation {
			case ReLU:
				newDefs = append(newDefs, LayerDef{Type: ReLU})
			case Sigmoid:
				newDefs = append(newDefs, LayerDef{Type: Sigmoid})
			case Tanh:
				newDefs = append(newDefs, LayerDef{Type: Tanh})
			case Maxout:
				groupSize := 2
				if def.Maxout != nil {
					groupSize = def.Maxout.GroupSize
				}
				newDefs = append(newDefs, LayerDef{
					Type: Maxout,
					LayerConfig: MaxoutLayerConfig{
						GroupSize: groupSize,
					},
				})
			default:
				panic("unsupported activation")
			}
		}

		// Add dropout layer
		if def.Dropout != nil {
			newDefs = append(newDefs, LayerDef{
				Type:        Dropout,
				LayerConfig: def.Dropout,
			})
		}
	}
	return newDefs
}
