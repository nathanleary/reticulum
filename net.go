package reticulum

import (
	"errors"

	layers "github.com/nathanleary/reticulum/layers"
	volume "github.com/nathanleary/reticulum/volume"
)

const (
	// DefaultDropout is the default dropout rate of 0.5 or 50%. Everything less than the dropout rate will be dropped.
	DefaultDropout float64 = 0.5
)

// Network is the neural network interface.
type Network interface {
	Size() int
	Layers() []layers.Layer

	Forward(vol *volume.Volume, training bool) *volume.Volume
	Backward(index int) float64
	GetCostLoss(vol *volume.Volume, index int) float64

	// GetPrediction assumes the last layer in the network is a SoftMax layer.
	GetPrediction() int
	GetResponse() []layers.LayerResponse

	MultiDimensionalLoss(losses []float64) float64
	DimensionalLoss(index int, value float64) float64
}

// NewNetwork creates a new network from the layer definitions
func NewNetwork(defs []layers.LayerDef) (Network, error) {
	if len(defs) <= 2 {
		return nil, errors.New("at least one input and one loss layer are required")
	} else if defs[0].Type != layers.Input {
		return nil, errors.New("first layer must be the input layer, to declare size of inputs")
	}

	// Add activation layers
	defs = layers.ActivateLayers(defs)

	var newLayers []layers.Layer
	for i, def := range defs {
		if i > 0 {
			prev := defs[i-1]
			def.Input = prev.Output
		}

		switch def.Type {
		case layers.FullyConnected:
			newLayers = append(newLayers, layers.NewFullyConnectedLayer(def))
		case layers.Dropout:
			newLayers = append(newLayers, layers.NewDropoutLayer(def))
		case layers.Input:
			newLayers = append(newLayers, layers.NewInputLayer(def))
		case layers.SoftMax:
			newLayers = append(newLayers, layers.NewSoftmaxLayer(def))
		case layers.Regression:
			newLayers = append(newLayers, layers.NewRegressionLayer(def))
		case layers.Conv:
			newLayers = append(newLayers, layers.NewConvLayer(def))
		case layers.Pool:
			newLayers = append(newLayers, layers.NewPoolLayer(def))
		case layers.ReLU:
			newLayers = append(newLayers, layers.NewReluLayer(def))
		case layers.Sigmoid:
			newLayers = append(newLayers, layers.NewSigmoidLayer(def))
		case layers.Tanh:
			newLayers = append(newLayers, layers.NewTanhLayer(def))
		case layers.Maxout:
			newLayers = append(newLayers, layers.NewMaxoutLayer(def))
		case layers.SVM:
			newLayers = append(newLayers, layers.NewSVMLayer(def))
		// case layers.LocalResponseNorm:
		default:
			return nil, errors.New("unrecognized layer type")
		}
	}
	return &network{newLayers}, nil
}

type network struct {
	layers []layers.Layer
}

func (n *network) Size() int {
	return len(n.layers)
}

func (n *network) Layers() []layers.Layer {
	return n.layers
}

func (n *network) Forward(vol *volume.Volume, training bool) *volume.Volume {
	actions := n.layers[0].Forward(vol, training)
	for index := 1; index < len(n.layers); index++ {
		actions = n.layers[index].Forward(vol, training)
	}
	return actions
}

func (n *network) Backward(index int) float64 {
	size := n.Size()

	// Calculate loss
	lossLayer, ok := n.layers[size-1].(layers.LossLayer)
	if !ok {
		panic("expecting loss layer as last layer in network")
	}
	loss := lossLayer.Loss(index)

	// Propogate backwards
	for index := n.Size() - 2; index >= 0; index-- {
		n.layers[index].Backward()
	}
	return loss
}

func (n *network) GetCostLoss(vol *volume.Volume, index int) float64 {
	n.Forward(vol, false)

	// Calculate loss
	lossLayer, ok := n.layers[n.Size()-1].(layers.LossLayer)
	if !ok {
		panic("expecting loss layer as last layer in network")
	}
	return lossLayer.Loss(index)
}

func (n *network) GetPrediction() int {
	// this is a convenience function for returning the argmax
	// prediction, assuming the last layer of the net is a softmax
	S := n.layers[n.Size()-1]
	if S.Type() != layers.SoftMax {
		panic("GetPrediction assumes Softmax is the last layer in the network")
	}
	return layers.GetSoftMaxPrediction(S)
}

func (n *network) GetResponse() []layers.LayerResponse {
	// accumulate parameters and gradients for the entire network
	resp := []layers.LayerResponse{}
	for index := 0; index < len(n.layers); index++ {
		layerResponse := n.layers[index].GetResponse()
		resp = append(resp, layerResponse...)
	}
	return resp
}

// MultiDimensionalLoss computes the total loss for each of the values given.
func (n *network) MultiDimensionalLoss(y []float64) float64 {
	lossLayer, ok := n.layers[n.Size()-1].(layers.RegressionLossLayer)
	if !ok {
		panic("MultiDimensionalLoss assumes a Regression layer is the last layer in the network")
	}
	return lossLayer.MultiDimensionalLoss(y)
}

func (n *network) DimensionalLoss(index int, value float64) float64 {
	lossLayer, ok := n.layers[n.Size()-1].(layers.RegressionLossLayer)
	if !ok {
		panic("DimensionalLoss assumes a Regression layer is the last layer in the network")
	}
	return lossLayer.DimensionalLoss(index, value)
}
