// What it does:
//
// This example detects motion using a delta threshold from the first frame,
// and then finds contours to determine where the object is located.
//
// Very loosely based on Adrian Rosebrock code located at:
// http://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/
//
// How to run:
//
// 		go run ./cmd/motion-detect/main.go 0
//

package main

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"

	"github.com/danhigham/gocv-blob/blob"
	uuid "github.com/satori/go.uuid"
	"gocv.io/x/gocv"
	"robpike.io/filter"
)

const minimumArea = 3000
const pixelDistance = 115

// const actualDistanceMilli = 14630
const fov = 112

// const distance_to_road = 90.5 // distance to road in mm
const distance_to_road = 49.5
const image_width = 640.0

type BackgroundMask struct {
	mask []gocv.Mat
}

func degToRad(degrees float64) float64 {
	return degrees * math.Pi / 180
}

func NewBackgroundMask(filename string) (*BackgroundMask, error) {
	img := gocv.IMRead(filename, gocv.IMReadColor)
	if img.Empty() {
		return &BackgroundMask{}, errors.New(fmt.Sprintf("Error reading image from: %v", filename))
	}

	bm := &BackgroundMask{
		mask: gocv.Split(img),
	}
	return bm, nil
}

func (bm BackgroundMask) isInsideMask(c []image.Point) bool {
	rect := gocv.BoundingRect(c)
	center := image.Pt((rect.Min.X*2+rect.Dx())/2, (rect.Min.Y*2+rect.Dy())/2)

	maskR := bm.mask[0].GetUCharAt(center.Y, center.X)
	maskG := bm.mask[1].GetUCharAt(center.Y, center.X)
	maskB := bm.mask[2].GetUCharAt(center.Y, center.X)

	return (maskR + maskG + maskB) > 0
}

func isTrackable(c []image.Point) bool {
	area := gocv.ContourArea(c)
	return !(area < minimumArea)
}

func getBoundingBoxes(contours [][]image.Point) []image.Rectangle {
	var rects []image.Rectangle
	for _, c := range contours {
		rects = append(rects, gocv.BoundingRect(c))
	}
	return rects
}

func writeMatToFile(mat gocv.Mat, filename string) {
	target, err := mat.ToImage()
	if err != nil {
		panic(err)
	}
	f, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	jpeg.Encode(f, target, nil)
}

func loadImage(filename string) ([]gocv.Mat, error) {
	img := gocv.IMRead(filename, gocv.IMReadColor)
	if img.Empty() {
		return []gocv.Mat{}, errors.New(fmt.Sprintf("Error reading image from: %v", filename))
	}

	mat := gocv.Split(img)
	return mat, nil
}

func distanceBetweenPoints(p1 image.Point, p2 image.Point) float64 {
	intX := math.Abs(float64(p1.X - p2.X))
	intY := math.Abs(float64(p1.Y - p2.Y))
	return math.Sqrt(math.Pow(intX, 2) + math.Pow(intY, 2))
}

func uuidin(a uuid.UUID, list []uuid.UUID) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

func padRect(rect image.Rectangle, padAmount int) image.Rectangle {
	min := rect.Min
	max := rect.Max

	min.X -= padAmount
	min.Y -= padAmount

	max.X += padAmount
	max.Y += padAmount
	if min.X < 0 {
		min.X = 0
	}
	if min.Y < 0 {
		min.Y = 0
	}
	if max.X > 639 {
		max.X = 639
	}
	if max.Y > 479 {
		max.X = 479
	}

	return image.Rectangle{Min: min, Max: max}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("How to run:\n\tmotion-detect [camera ID]")
		return
	}

	bm, err := NewBackgroundMask("./background_mask.jpg")
	if err != nil {
		fmt.Printf("Error opening background mask - %s", err)
		return
	}

	frame_width := 2 * (math.Tan(degToRad(fov*0.5)) * distance_to_road)
	ftperpixel := frame_width / image_width

	// create centroid tracker
	// tracker := blob.NewCentroidTrackerDefaults()
	tracker := blob.NewCentroidTracker(10, 40, 20)

	// parse args
	streamURL := os.Args[1]

	webcam, err := gocv.VideoCaptureFile(streamURL)
	if err != nil {
		fmt.Printf("Error opening video capture streamURL: %v\n", streamURL)
		return
	}
	defer webcam.Close()

	// windowHeight := 400

	window := gocv.NewWindow("Motion Window")
	defer window.Close()

	// window2 := gocv.NewWindow("Blobs")
	// defer window2.Close()

	img := gocv.NewMat()
	defer img.Close()

	imgDelta := gocv.NewMat()
	defer imgDelta.Close()

	imgThresh := gocv.NewMat()
	defer imgThresh.Close()

	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close()

	// if ok := webcam.Read(&img); !ok {
	// 	fmt.Printf("Stream closed: %v\n", streamURL)
	// 	return
	// }
	//
	// writeMatToFile(img, "/tmp/background.jpg")

	fmt.Printf("Start reading stream: %v\n", streamURL)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Stream closed: %v\n", streamURL)
			return
		}
		if img.Empty() {
			continue
		}

		// first phase of cleaning up image, obtain foreground only
		mog2.Apply(img, &imgDelta)

		// remaining cleanup of the image to use for finding contours.
		// first use threshold
		gocv.Threshold(imgDelta, &imgThresh, 25, 255, gocv.ThresholdBinary)

		gocv.MedianBlur(imgThresh, &imgThresh, 7)

		// kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(10, 10))
		// defer kernel.Close()
		// gocv.Dilate(imgThresh, &imgThresh, kernel)

		// now find contours
		contours := gocv.FindContours(imgThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
		contours = filter.Choose(contours, isTrackable).([][]image.Point)
		contours = filter.Choose(contours, bm.isInsideMask).([][]image.Point)
		bb := getBoundingBoxes(contours)

		tracker.Update(bb)

		if len(tracker.NewObjects) > 0 {
			fmt.Printf("New: %+v\n", tracker.NewObjects)
			fmt.Printf("Obj: %+v\n", tracker.Objects)
		}

		for i, o := range tracker.Objects {
			gocv.Rectangle(&img, o.CurrentRect, color.RGBA{0, 0, 255, 0}, 1)
			last := o.GetLastPoint()
			first := o.Track[0]
			gocv.Line(&img, first.Point, last.Point, color.RGBA{0, 255, 0, 0}, 1)
			statusColor := color.RGBA{0, 255, 0, 0}

			distance, duration := o.SpaceTimeTravelled()

			mph := ((distance * ftperpixel) / duration.Seconds()) * 0.681818

			if len(tracker.NewObjects) > 0 {
				if uuidin(i, tracker.NewObjects) {
					fmt.Printf("CurrentRect: %+v\n", o.CurrentRect)
					carFrame := img.Region(padRect(o.CurrentRect, 20))
					carMat := carFrame.Clone()
					defer carMat.Close()
					defer carFrame.Close()

					gocv.PutText(&carMat, fmt.Sprintf("%3.2f", mph), image.Pt(2, 2), gocv.FontHersheyPlain, 1, statusColor, 1)
					writeMatToFile(carMat, fmt.Sprintf("./cars/%s.jpg", i.String()[0:7]))
				}
			}

			gocv.PutText(&img, fmt.Sprintf("%3.2f mph (%2.1f)", mph, duration.Seconds()), image.Pt(o.Center.X, o.Center.Y), gocv.FontHersheyPlain, 1.2, statusColor, 1)
		}

		window.IMShow(img)
		// window2.IMShow(imgThresh)

		if window.WaitKey(1) == 27 {
			break
		}
	}
}
