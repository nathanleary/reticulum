package layers

import (
	"fmt"
	"math"

	"github.com/nathanleary/reticulum/volume"
)

// NewPoolLayerConfig creates a new poolLayer config with the given options.
func NewPoolLayerConfig(filters int, opts ...LayerOptionFunc) LayerConfig {
	if filters <= 0 {
		panic("Filter count must be greater than 0")
	}

	conf := &poolLayerConfig{
		Sx:      filters,
		Sy:      filters,
		Stride:  2,
		Padding: 0,
	}
	for i := 0; i < len(opts); i++ {
		err := opts[i](conf)
		if err != nil {
			panic(err)
		}
	}
	return conf
}

type poolLayerConfig struct {
	Sx      int
	Sy      int
	Stride  int
	Padding int
}

// NewPoolLayer creates a new pool layer.
func NewPoolLayer(def LayerDef) Layer {

	// Validate input
	if def.Type != Pool {
		panic(fmt.Errorf("Invalid layer type: %s != pool", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for pool layer"))
	} else if def.LayerConfig == nil {
		panic(fmt.Errorf("Config cannot be nil for pool layer"))
	}

	// Get config
	conf, ok := def.LayerConfig.(*poolLayerConfig)
	if !ok {
		panic("Invalid LayerConfig for PoolLayer")
	}

	// Set Sy
	if conf.Sy <= 0 {
		conf.Sy = conf.Sx
	}

	// Output dimensions
	outDepth := def.Input.Z
	outSx := math.Floor((float64(def.Input.X)+float64(conf.Padding)*2.0-float64(conf.Sx))/float64(conf.Stride) + 1)
	outSy := math.Floor((float64(def.Input.Y)+float64(conf.Padding)*2.0-float64(conf.Sy))/float64(conf.Stride) + 1)
	outDim := volume.NewDimensions(int(outSx), int(outSy), outDepth)

	return &poolLayer{conf, def.Input, outDim, nil, nil, make([]int, outDim.Size()), make([]int, outDim.Size())}
}

type poolLayer struct {
	conf   *poolLayerConfig
	input  volume.Dimensions
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	switchX []int
	switchY []int
}

func (*poolLayer) Type() LayerType {
	return Pool
}

func (l *poolLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	A := volume.NewVolume(l.output, volume.WithZeros())

	var n int
	for d := 0; d < l.output.Z; d++ {
		x := -l.conf.Padding
		for ax := 0; ax < l.output.X; ax++ {
			x += l.conf.Stride
			y := -l.conf.Padding
			for ay := 0; ay < l.output.Y; ay++ {
				y += l.conf.Stride

				// convolve centered at this particular location
				a := -1e5
				winX, winY := -1, -1
				for fx := 0; fx < l.conf.Sx; fx++ {
					for fy := 0; fy < l.conf.Sy; fy++ {
						oy := y + fy
						ox := x + fx
						if oy >= 0 && oy < l.input.Y && ox >= 0 && ox < l.input.X {
							v := l.inVol.Get(ox, oy, d)
							// perform max pooling and store pointers to where
							// the max came from. This will speed up backprop
							// and can help make nice visualizations in future
							if v > a {
								a = v
								winX = ox
								winY = oy
							}
						}
					}
				}
				l.switchX[n] = winX
				l.switchY[n] = winY
				n++
				A.Set(ax, ay, d, a)
			}
		}
	}

	l.outVol = A
	return l.outVol
}

func (l *poolLayer) Backward() {
	l.inVol.ZeroGrad()

	var n int
	for d := 0; d < l.output.Z; d++ {
		x := -l.conf.Padding
		for ax := 0; ax < l.output.X; ax++ {
			x += l.conf.Stride
			y := -l.conf.Padding
			for ay := 0; ay < l.output.Y; ay++ {
				y += l.conf.Stride
				chainGrad := l.outVol.GetGrad(ax, ay, d)
				l.inVol.AddGrad(l.switchX[n], l.switchY[n], d, chainGrad)
				n++
			}
		}
	}
}

func (l *poolLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
