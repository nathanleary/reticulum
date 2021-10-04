package layers

import (
	"fmt"
	"math"

	"github.com/nathanleary/reticulum/volume"
)

// NewTanhLayer creates a new ReLU (rectified linear unit) layer.
func NewTanhLayer(def LayerDef) Layer {
	if def.Type != Tanh {
		panic(fmt.Errorf("Invalid layer type: %s != tanh", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for tanh layer"))
	}
	return &tanhLayer{def.Output, nil, nil}
}

type tanhLayer struct {
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (l *tanhLayer) Type() LayerType {
	return Tanh
}

func (l *tanhLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	v2 := vol.CloneAndZero()

	n := vol.Size()
	for i := 0; i < n; i++ {
		v2.SetByIndex(i, math.Tanh(vol.GetByIndex(i)))
	}

	l.outVol = v2
	return l.outVol
}

func (l *tanhLayer) Backward() {
	n := l.inVol.Size()
	l.inVol.ZeroGrad()

	for i := 0; i < n; i++ {
		v2wi := l.outVol.GetByIndex(i)
		l.inVol.SetGradByIndex(i, (1.0-v2wi*v2wi)*l.outVol.GetGradByIndex(i))
	}
}

func (*tanhLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
