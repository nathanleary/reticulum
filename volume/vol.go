package volume

import (
	"math"
	"math/rand"
)

// NewDimensions creates a new Dimension struct
func NewDimensions(x, y, z int) Dimensions {
	return Dimensions{x, y, z}
}

// Dimensions represents the volumetric size of the data.
type Dimensions struct {
	X, Y, Z int
}

// Size returns the number of elements.
func (d *Dimensions) Size() int {
	return d.X * d.Y * d.Z
}

// Clone returns a new Dimentsions struct with the same dimensions.
func (d *Dimensions) Clone() Dimensions {
	return Dimensions{d.X, d.Y, d.Z}
}

// Options stores volume options
type Options struct {
	Zero            bool
	HasInitialValue bool
	InitialValue    float64
	Weights         []float64
}

// OptionFunc modifies the Options when creating a new Volume.
type OptionFunc func(*Options)

// WithInitialValue sets the initial values of the Volume.
func WithInitialValue(value float64) OptionFunc {
	return func(opts *Options) {
		opts.HasInitialValue = true
		opts.InitialValue = value
	}
}

// WithZeros sets the initial values of the Volume to zero.
func WithZeros() OptionFunc {
	return func(opts *Options) {
		opts.HasInitialValue = true
		opts.Zero = true
	}
}

// WithWeights initializes the Volume with the given weights.
func WithWeights(w []float64) OptionFunc {
	return func(opts *Options) {
		opts.Weights = w
	}
}

// NewVolume creates a new Volume of the given size and options.
func NewVolume(dim Dimensions, optFuncs ...OptionFunc) *Volume {
	n := dim.Size()
	w := make([]float64, n, n)
	dw := make([]float64, n, n)

	// Update opts
	opts := &Options{}
	for _, optFn := range optFuncs {
		optFn(opts)
	}

	// Initialize weights
	if opts.HasInitialValue {
		if !opts.Zero {
			for i := 0; i < n; i++ {
				w[i] = opts.InitialValue
			}
		} else {
			// Arrays already contain zeros.
		}
	} else if opts.Weights != nil {
		if len(opts.Weights) != dim.Z {
			panic("Invalid input weights: depth inconsistencies")
		} else if dim.X != 1 {
			panic("Invalid volume dimensions: X must equal 1 when weights are given")
		} else if dim.Y != 1 {
			panic("Invalid volume dimensions: Y must equal 1 when weights are given")
		}
		// Copy weights
		copy(w, opts.Weights)
	} else {

		// weight normalization is done to equalize the output
		// variance of every neuron, otherwise neurons with a lot
		// of incoming connections have outputs of larger variance
		desiredStdDev := math.Sqrt(1.0 / float64(n))
		for i := 0; i < n; i++ {

			// Gaussian distribution with a mean of 0 and the given stdev
			w[i] = rand.NormFloat64() * desiredStdDev
		}
	}

	return &Volume{
		dim, w, dw,
	}
}

// Volume is the basic building block of all the data in a network.
// It is essentially a 3D block of numbers with a width (sx), height (sy),
// and a depth (depth). It is used to hold data for all the filters, volumes,
// weights and gradients w.r.t. the data.
type Volume struct {
	dim Dimensions
	w   []float64
	dw  []float64
}

// Dimensions returns the Dimensions of the Volume.
func (v *Volume) Dimensions() Dimensions {
	return v.dim
}

// Size returns the number of elements.
func (v *Volume) Size() int {
	return v.dim.Size()
}

// getIndex returns the array index for the given position.
func (v *Volume) getIndex(x, y, d int) int {
	return ((v.dim.X*y)+x)*v.dim.Z + d
}

// Get returns a weight for the given position in the Volume.
func (v *Volume) Get(x, y, d int) float64 {
	return v.w[v.getIndex(x, y, d)]
}

// Set updates the position in the Volume.
func (v *Volume) Set(x, y, d int, val float64) {
	v.w[v.getIndex(x, y, d)] = val
}

// GetByIndex returns a weight for the given index in the Volume.
func (v *Volume) GetByIndex(index int) float64 {
	return v.w[index]
}

// SetByIndex updates the position in the Volume by index.
func (v *Volume) SetByIndex(index int, val float64) {
	v.w[index] = val
}

// Add adds the given value to the weight for the given position.
func (v *Volume) Add(x, y, d int, val float64) {
	v.w[v.getIndex(x, y, d)] += val
}

// Mult multiplies the given value to the weight for the given position.
func (v *Volume) Mult(x, y, d int, val float64) {
	v.w[v.getIndex(x, y, d)] *= val
}

// MultByIndex multiplies the given value to the weight for the given index position.
func (v *Volume) MultByIndex(index int, val float64) {
	v.w[index] *= val
}

// GetGrad returns the gradient at the given position.
func (v *Volume) GetGrad(x, y, d int) float64 {
	return v.dw[v.getIndex(x, y, d)]
}

// SetGrad updates the gradient at the given position.
func (v *Volume) SetGrad(x, y, d int, val float64) {
	v.dw[v.getIndex(x, y, d)] = val
}

// GetGradByIndex returns a gradient for the given index in the Volume.
func (v *Volume) GetGradByIndex(index int) float64 {
	return v.dw[index]
}

// SetGradByIndex updates the gradient position in the Volume by index.
func (v *Volume) SetGradByIndex(index int, val float64) {
	v.dw[index] = val
}

// AddGrad adds the given value to the gradient for the given position.
func (v *Volume) AddGrad(x, y, d int, val float64) {
	v.dw[v.getIndex(x, y, d)] += val
}

// AddGradByIndex adds the given value to the gradient for the given index.
func (v *Volume) AddGradByIndex(index int, val float64) {
	v.dw[index] += val
}

// Clone creates a new Volume with cloned weights and zeroed gradients.
func (v *Volume) Clone() *Volume {
	vol := NewVolume(v.dim, WithZeros())
	copy(vol.w, v.w)
	return vol
}

// CloneAndZero creates a Volume of the same size but with zero weights and gradients.
func (v *Volume) CloneAndZero() *Volume {
	return NewVolume(v.dim, WithZeros())
}

// AddFrom adds the weights from another Volume.
func (v *Volume) AddFrom(vol *Volume) {
	for i := 0; i < v.Size(); i++ {
		v.w[i] += vol.w[i]
	}
}

// AddFromScaled adds the weights from another Volume and scaled with the given value.
func (v *Volume) AddFromScaled(vol *Volume, scale float64) {
	for i := 0; i < v.Size(); i++ {
		v.w[i] += vol.w[i] * scale
	}
}

// ZeroGrad sets the gradients to 0.
func (v *Volume) ZeroGrad() {
	for i := 0; i < v.Size(); i++ {
		v.dw[i] = 0.0
	}
}

// SetConst sets the weights to the given value.
func (v *Volume) SetConst(val float64) {
	for i := 0; i < v.Size(); i++ {
		v.w[i] = val
	}
}

// Weights returns all the weights for the volume.
func (v *Volume) Weights() []float64 {
	return v.w
}

// Gradients returns all the gradients for the volume.
func (v *Volume) Gradients() []float64 {
	return v.dw
}
