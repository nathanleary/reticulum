package layers

import (
	"fmt"

	"github.com/nathanleary/reticulum/volume"
)

// MaxoutLayerConfig contains the maxout group size.
type MaxoutLayerConfig struct {
	GroupSize int
}

// NewMaxoutLayer creates a new maxout layer.
func NewMaxoutLayer(def LayerDef) Layer {
	if def.Type != Maxout {
		panic(fmt.Errorf("Invalid layer type: %s != maxout", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for maxout layer"))
	}

	// Cast layer config
	conf, ok := def.LayerConfig.(*MaxoutLayerConfig)
	if !ok {
		panic(fmt.Errorf("Invalid layer config: expected MaxoutLayerConfig got %T", conf))
	}

	// Validate group size
	if conf.GroupSize <= 0 {
		panic(fmt.Errorf("Group size cannot be  <= 0 for maxout layer"))
	}

	return &maxoutLayer{conf, def.Output, nil, nil, make([]int, def.Output.Size())}
}

type maxoutLayer struct {
	conf   *MaxoutLayerConfig
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	switches []int
}

func (l *maxoutLayer) Type() LayerType {
	return Maxout
}

func (l *maxoutLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {

	l.inVol = vol
	v2 := l.outVol.CloneAndZero()
	n := l.output.Z

	// optimization branch. If we're operating on 1D arrays we dont have
	// to worry about keeping track of x,y,d coordinates inside
	// input volumes. In convnets we do :(
	if l.output.X == 1 && l.output.Y == 1 {
		for i := 0; i < n; i++ {
			ix := i * l.conf.GroupSize
			a := l.inVol.GetByIndex(ix)

			var ai int
			for j := 1; j < l.conf.GroupSize; j++ {
				a2 := l.inVol.GetByIndex(ix + j)
				if a2 > a {
					a = a2
					ai = j
				}
			}
			v2.SetByIndex(i, a)
			l.switches[i] = ix + ai
		}
	} else {
		var si int
		for x := 0; x < l.output.X; x++ {
			for y := 0; y < l.output.Y; y++ {
				for i := 0; i < n; i++ {
					ix := i * l.conf.GroupSize
					a := l.inVol.Get(x, y, ix)

					var ai int
					for j := 1; j < l.conf.GroupSize; j++ {
						a2 := l.inVol.Get(x, y, ix+j)
						if a2 > a {
							a = a2
							ai = j
						}
					}
					v2.Set(x, y, i, a)
					l.switches[n] = ix + ai
					si++
				}
			}
		}
	}

	l.outVol = v2
	return l.outVol
}

func (l *maxoutLayer) Backward() {
	n := l.output.Z
	l.inVol.ZeroGrad()

	if l.output.X == 1 && l.output.Y == 1 {
		for i := 0; i < n; i++ {
			chainGrad := l.outVol.GetGradByIndex(i)
			l.inVol.SetGradByIndex(l.switches[i], chainGrad)
		}
	} else {

		// counter for switches
		var si int

		// bleh okay, lets do this the hard way
		for x := 0; x < l.output.X; x++ {
			for y := 0; y < l.output.Y; y++ {
				for i := 0; i < n; i++ {
					chainGrad := l.outVol.GetGrad(x, y, i)
					l.inVol.SetGrad(x, y, l.switches[si], chainGrad)
					si++
				}
			}
		}
	}
}

func (l *maxoutLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
