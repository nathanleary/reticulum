package volume

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func randArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < len(arr); i++ {
		arr[i] = rand.Float64()
	}
	return arr
}

func TestWithInitialValue(t *testing.T) {
	type test struct {
		name string
		want float64
		fn   OptionFunc
	}
	tests := []test{
		{"Initializing with Zero", 0.0, WithInitialValue(0.0)},
		{"Initializing with 0.75", 0.75, WithInitialValue(0.75)},
	}

	for i := 0; i < 5; i++ {
		v := rand.Float64()
		tests = append(tests, test{name: fmt.Sprintf("Initializing with random[%d] %v", i, v), want: v, fn: WithInitialValue(v)})
	}

	opts := &Options{}
	for _, tt := range tests {
		opts.HasInitialValue = false
		opts.InitialValue = -1.0
		opts.Zero = false
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(opts)

			if !reflect.DeepEqual(opts.HasInitialValue, true) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.HasInitialValue, true)
			}
			if !reflect.DeepEqual(opts.InitialValue, tt.want) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.InitialValue, tt.want)
			}
			if !reflect.DeepEqual(opts.Zero, false) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.Zero, false)
			}
		})
	}
}

func TestWithWeights_InvalidDepth(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic")
		}
	}()

	// This function should cause a panic
	NewVolume(Dimensions{1, 1, 10}, WithWeights(randArray(5)))
}

func TestWithWeights_InvalidSy(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic")
		}
	}()

	// This function should cause a panic
	NewVolume(Dimensions{1, 2, 5}, WithWeights(randArray(5)))
}

func TestWithWeights_InvalidSx(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic")
		}
	}()

	// This function should cause a panic
	NewVolume(Dimensions{2, 1, 5}, WithWeights(randArray(5)))
}

func TestWithWeights(t *testing.T) {
	wgts := randArray(5)
	vol := NewVolume(Dimensions{1, 1, 5}, WithWeights(wgts))
	if !reflect.DeepEqual(vol.w, wgts) {
		t.Errorf("WithWeights() = %v, want %v", vol.w, wgts)
	}
}

func TestWithZeros(t *testing.T) {
	tests := []struct {
		name string
		fn   OptionFunc
	}{
		{"With Zeros", WithZeros()},
	}
	opts := &Options{}
	for _, tt := range tests {
		opts.Zero = false
		opts.HasInitialValue = false
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(opts)

			if !reflect.DeepEqual(opts.HasInitialValue, true) {
				t.Errorf("WithZeros() = %v, want %v", opts.HasInitialValue, true)
			}
			if !reflect.DeepEqual(opts.InitialValue, 0.0) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.InitialValue, 0.0)
			}
			if !reflect.DeepEqual(opts.Zero, true) {
				t.Errorf("WithZeros() = %v, want %v", opts.Zero, true)
			}
		})
	}
}

func TestNewVolume(t *testing.T) {
	type args struct {
		dim      Dimensions
		optFuncs []OptionFunc
	}
	tests := []struct {
		name string
		args args
		want *Volume
	}{
		{"NewVolumeWithZeros", args{Dimensions{1, 1, 25}, []OptionFunc{WithZeros()}}, NewVolume(Dimensions{1, 1, 25}, WithZeros())},
		{"NewVolumeWithInitialValue", args{Dimensions{1, 1, 25}, []OptionFunc{WithInitialValue(0.5)}}, NewVolume(Dimensions{1, 1, 25}, WithInitialValue(0.5))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewVolume(tt.args.dim, tt.args.optFuncs...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewVolume(Dimensions{) = %v, want} %v", got, tt.want)
			}
		})
	}
}

func TestVolume_Get(t *testing.T) {
	dim := Dimensions{1, 2, 6}
	vol := NewVolume(dim)
	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				want := vol.w[vol.getIndex(x, y, d)]
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}
}

func TestVolume_Set(t *testing.T) {
	dim := Dimensions{1, 2, 6}
	vol := NewVolume(dim, WithZeros())
	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				value := rand.Float64()
				vol.Set(x, y, d, value)
				want := vol.w[vol.getIndex(x, y, d)]

				if value != want {
					t.Errorf("Volume.Set() = %v, want %v", value, want)
				}
			}
		}
	}
}

func TestVolume_Add(t *testing.T) {
	type fields struct {
		dim Dimensions
		w   []float64
		dw  []float64
	}
	type args struct {
		x   int
		y   int
		d   int
		val float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{"Zeros", fields{Dimensions{1, 2, 6}, make([]float64, 12), make([]float64, 12)}, args{1, 0, 5, 0.5}},
		{"Random", fields{Dimensions{1, 2, 6}, randArray(12), randArray(12)}, args{1, 0, 5, 0.1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				dim: tt.fields.dim,
				w:   tt.fields.w,
				dw:  tt.fields.dw,
			}
			before := v.Get(tt.args.x, tt.args.y, tt.args.d)
			v.Add(tt.args.x, tt.args.y, tt.args.d, tt.args.val)

			value := v.Get(tt.args.x, tt.args.y, tt.args.d)
			want := before + tt.args.val
			if value != want {
				t.Errorf("Volume.Add() = %v, want %v", value, want)
			}
		})
	}
}

func TestVolume_getIndex(t *testing.T) {
	type args struct {
		x int
		y int
		d int
	}
	tests := []struct {
		name string
		args args
		vol  *Volume
		want int
	}{
		{"Zero XY", args{0, 0, 5}, NewVolume(Dimensions{1, 1, 20}, WithZeros()), 5},
		{"Zero Y", args{1, 0, 5}, NewVolume(Dimensions{1, 1, 20}, WithZeros()), 25},
		{"Zero X", args{0, 1, 5}, NewVolume(Dimensions{1, 1, 20}, WithZeros()), 25},
		{"Zero D", args{1, 1, 0}, NewVolume(Dimensions{1, 20, 5}, WithZeros()), 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vol
			if got := v.getIndex(tt.args.x, tt.args.y, tt.args.d); got != tt.want {
				t.Errorf("Volume.getIndex() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_GetGrad(t *testing.T) {
	type fields struct {
		dim Dimensions
		w   []float64
		dw  []float64
	}
	tests := []struct {
		name   string
		fields fields
	}{
		// TODO: Add test cases.
		{"GetGrad_Random", fields{Dimensions{1, 2, 20}, randArray(40), randArray(40)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vol := &Volume{
				dim: tt.fields.dim,
				w:   tt.fields.w,
				dw:  tt.fields.dw,
			}
			dim := tt.fields.dim
			for x := 0; x < dim.X; x++ {
				for y := 0; y < dim.Y; y++ {
					for d := 0; d < dim.Z; d++ {
						want := vol.dw[vol.getIndex(x, y, d)]
						if got := vol.GetGrad(x, y, d); got != want {
							t.Errorf("Volume.GetGrad() = %v, want %v", got, want)
						}
					}
				}
			}
		})
	}
}

func TestVolume_SetGrad(t *testing.T) {
	type fields struct {
		dim Dimensions
		w   []float64
		dw  []float64
	}
	tests := []struct {
		name   string
		fields fields
		want   float64
	}{
		// TODO: Add test cases.
		{"SetGrad_Random", fields{Dimensions{1, 2, 20}, randArray(40), randArray(40)}, 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vol := &Volume{
				dim: tt.fields.dim,
				w:   tt.fields.w,
				dw:  tt.fields.dw,
			}

			dim := tt.fields.dim
			for x := 0; x < dim.X; x++ {
				for y := 0; y < dim.Y; y++ {
					for d := 0; d < dim.Z; d++ {
						vol.SetGrad(x, y, d, tt.want)
						if got := vol.GetGrad(x, y, d); got != tt.want {
							t.Errorf("Volume.SetGrad() = %v, want %v", got, tt.want)
						}
					}
				}
			}
		})
	}
}

func TestVolume_AddGrad(t *testing.T) {
	type fields struct {
		dim Dimensions
		w   []float64
		dw  []float64
	}
	tests := []struct {
		name   string
		fields fields
		arg    float64
	}{
		// TODO: Add test cases.
		{"AddGrad_Random", fields{Dimensions{1, 2, 20}, randArray(40), randArray(40)}, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vol := &Volume{
				dim: tt.fields.dim,
				w:   tt.fields.w,
				dw:  tt.fields.dw,
			}

			dim := tt.fields.dim
			for x := 0; x < dim.X; x++ {
				for y := 0; y < dim.Y; y++ {
					for d := 0; d < dim.Z; d++ {
						want := vol.dw[vol.getIndex(x, y, d)] + tt.arg
						vol.AddGrad(x, y, d, tt.arg)
						if got := vol.GetGrad(x, y, d); got != want {
							t.Errorf("Volume.SetGrad() = %v, want %v", got, want)
						}
					}
				}
			}

		})
	}
}

func TestVolume_Clone(t *testing.T) {
	v := NewVolume(Dimensions{1, 2, 6})
	if got := v.Clone(); !reflect.DeepEqual(got, v) {
		t.Errorf("Volume.Clone() = %v, want %v", got, v)
	}
}

func TestVolume_CloneAndZero(t *testing.T) {
	v := NewVolume(Dimensions{1, 2, 6}, WithZeros())
	if got := v.CloneAndZero(); !reflect.DeepEqual(got, v) {
		t.Errorf("Volume.CloneAndZero() = %v, want %v", got, v)
	}
}

func TestVolume_AddFrom(t *testing.T) {
	dim := Dimensions{1, 2, 20}
	vol := NewVolume(dim)
	vol2 := NewVolume(dim)
	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				ix := vol.getIndex(x, y, d)
				want := vol.w[ix] + vol2.w[ix]
				vol.AddFrom(vol2)
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.AddFrom() = %v, want %v", got, want)
				}
			}
		}
	}
}

func TestVolume_AddFromScaled(t *testing.T) {

	scale := 0.5
	dim := Dimensions{1, 2, 20}
	vol := NewVolume(dim)
	vol2 := NewVolume(dim)
	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				ix := vol.getIndex(x, y, d)
				want := vol.w[ix] + vol2.w[ix]*scale
				vol.AddFromScaled(vol2, scale)
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.AddFromScaled() = %v, want %v", got, want)
				}
			}
		}
	}
}

func TestVolume_SetConst(t *testing.T) {
	want := -1.0
	dim := Dimensions{1, 2, 6}
	vol := NewVolume(dim, WithInitialValue(-1.0))

	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}

	want = 5.0
	vol.SetConst(want)

	for x := 0; x < dim.X; x++ {
		for y := 0; y < dim.Y; y++ {
			for d := 0; d < dim.Z; d++ {
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}
}
