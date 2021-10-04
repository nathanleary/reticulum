package layers

import (
	"fmt"

	"github.com/nathanleary/reticulum/volume"
)

// NewInputLayer creates a new input layer.
func NewInputLayer(def LayerDef) Layer {
	if def.Type != Input {
		panic(fmt.Errorf("Invalid layer type: %s != input", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for input layer"))
	}
	return &inputLayer{def.Output, nil, nil}
}

type inputLayer struct {
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (il *inputLayer) Type() LayerType {
	return Input
}

func (il *inputLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	il.inVol = vol
	il.outVol = vol
	return il.outVol
}

func (il *inputLayer) Backward() {
	panic(fmt.Errorf("Unsupported operation"))
}

func (il *inputLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
