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
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/danhigham/gocv-blob/blob"
	"github.com/hybridgroup/mjpeg"
	uuid "github.com/satori/go.uuid"
	"github.com/streadway/amqp"
	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
	"robpike.io/filter"
)

const minimumArea = 3000

// const actualDistanceMilli = 14630
const fov = 112

// const distance_to_road = 90.5 // distance to road in mm
const distance_to_road = 49.5
const image_width = 640.0

type CamStream struct {
	Stream  *mjpeg.Stream
	Channel chan gocv.Mat
}

type CarRegister map[uuid.UUID]*Car

type Car struct {
	Track   []CarTrack
	Tracker contrib.Tracker
}

type CarTrack struct {
	TrackPoint blob.TrackPoint
	Mat        *gocv.Mat
}

type CarMessage struct {
	ImageURI  string
	Speed     float64
	Distance  float64
	TimeStamp time.Time
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
	}
}

func (c *Car) MiddleMat() (*gocv.Mat, error) {
	if len(c.Track) == 0 {
		return nil, errors.New("Track length is zero!")
	}

	midPoint := c.Track[(len(c.Track) / 2)]
	return midPoint.Mat, nil
}

func (c *Car) SpaceTimeTravelled() (float64, time.Duration, error) {

	if c.Track == nil {
		return 0, 0, errors.New("Track is null!")
	}

	if len(c.Track) == 0 {
		return 0, 0, errors.New("Track length is zero!")
	}

	lastPoint := c.Track[len(c.Track)-1].TrackPoint
	firstPoint := c.Track[0].TrackPoint

	distance := 0.0

	for i := 0; i < len(c.Track)-2; i++ {
		distance += distanceBetweenPoints(c.Track[i].TrackPoint.Point, c.Track[i+1].TrackPoint.Point)
	}

	timeTaken := lastPoint.Created.Sub(firstPoint.Created)
	return distance, timeTaken, nil
}

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

func writeMatToBytes(mat *gocv.Mat) ([]byte, error) {
	var buf []byte
	target, err := mat.ToImage()
	if err != nil {
		return buf, err
	}
	b := bytes.NewBuffer(buf)
	err = jpeg.Encode(b, target, nil)
	if err != nil {
		return buf, err
	}
	return buf, nil
}

func writeMatToFile(mat *gocv.Mat, filename string) {
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

func removeCar(carMessageChan chan CarMessage, register CarRegister, id uuid.UUID) {

	car := register[id]

	distance, duration, err := car.SpaceTimeTravelled()
	mat, err := car.MiddleMat()

	if err == nil {

		frame_width := 2 * (math.Tan(degToRad(fov*0.5)) * distance_to_road)
		ftperpixel := frame_width / image_width
		ft := distance * ftperpixel

		if ft >= 60 { // need more than 60ft of distance for a good read

			mph := (ft / duration.Seconds()) * 0.681818

			fmt.Printf("%s Avg Speed: %3.2f mph across %3.2f ft\n", id.String(), mph, ft)
			fmt.Printf("Removing %s\n", id.String())

			s3Key := os.Getenv("S3_KEY")
			s3Secret := os.Getenv("S3_SECRET")
			s3Host := os.Getenv("S3_HOST")
			s3Bucket := os.Getenv("S3_BUCKET")

			s3Config := &aws.Config{
				Credentials:      credentials.NewStaticCredentials(s3Key, s3Secret, ""),
				Endpoint:         aws.String(s3Host),
				Region:           aws.String("us-east-1"),
				DisableSSL:       aws.Bool(false),
				S3ForcePathStyle: aws.Bool(true),
			}
			session := session.New(s3Config)
			s3Client := s3.New(session)

			clone := mat.Clone()
			defer clone.Close()
			matBytes, err := gocv.IMEncode(".jpg", clone)

			key := aws.String(fmt.Sprintf("%s.jpg", id.String()))

			_, err = s3Client.PutObject(&s3.PutObjectInput{
				Body:   bytes.NewReader(matBytes),
				Bucket: aws.String(s3Bucket),
				Key:    key,
			})
			if err != nil {
				fmt.Printf("Failed to upload data to %s/%s, %s\n", s3Bucket, *key, err.Error())
			}

			msg := CarMessage{
				ImageURI:  *key,
				Speed:     mph,
				Distance:  ft,
				TimeStamp: time.Now(),
			}

			carMessageChan <- msg

			// writeMatToFile(mat, fmt.Sprintf("./cars/%s.jpg", id.String()))
		}
	}

	delete(register, id)
}

func capture(camStream CamStream) {
	for {
		m := <-camStream.Channel
		buf, _ := gocv.IMEncode(".jpg", m)
		camStream.Stream.UpdateJPEG(buf)
	}

}

func openbrowser(url string) {
	var err error

	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		log.Fatal(err)
	}
}

var showWindowsFlag bool

func main() {
	flag.BoolVar(&showWindowsFlag, "show-windows", false, "Show windows for output preview")
	flag.Parse()

	// get env vars
	streamURL := os.Getenv("STREAM_URL")

	bm, err := NewBackgroundMask("./background_mask.jpg")
	if err != nil {
		fmt.Printf("Error opening background mask - %s", err)
		return
	}

	// start thread listening for car messages
	carMessageChan := make(chan CarMessage)

	go func() {
		rabbitURL := fmt.Sprintf("amqp://%s:%s@%s:5672/", os.Getenv("RABBIT_USER"), os.Getenv("RABBIT_PASS"), os.Getenv("RABBIT_HOST"))
		fmt.Printf("Connecting to AMPQ at %s\n", rabbitURL)

		conn, err := amqp.Dial(rabbitURL)
		failOnError(err, "Failed to connect to RabbitMQ")
		defer conn.Close()

		ch, err := conn.Channel()
		failOnError(err, "Failed to open a channel")
		defer ch.Close()

		q, err := ch.QueueDeclare(
			"cars", // name
			false,  // durable
			false,  // delete when unused
			false,  // exclusive
			false,  // no-wait
			nil,    // arguments
		)

		failOnError(err, "Failed to declare a queue")

		for {
			carMessage := <-carMessageChan

			jsonMsg, err := json.Marshal(carMessage)
			failOnError(err, "Failed to marshal json message")

			fmt.Printf("Publishing message %s\n", string(jsonMsg))

			err = ch.Publish(
				"",     // exchange
				q.Name, // routing key
				false,  // mandatory
				false,  // immediate
				amqp.Publishing{
					ContentType: "application/json",
					Body:        jsonMsg,
				})
		}

	}()

	trackingStream := CamStream{Stream: mjpeg.NewStream(), Channel: make(chan gocv.Mat)}

	go func() {
		http.Handle("/stream", trackingStream.Stream)
		log.Fatal(http.ListenAndServe("0.0.0.0:8080", nil))
	}()
	go capture(trackingStream)

	openbrowser("http://localhost:8080/stream")

	cars := make(CarRegister)

	// create centroid tracker
	// tracker := blob.NewCentroidTrackerDefaults()
	tracker := blob.NewCentroidTracker(20, 40, 10)

	webcam, err := gocv.VideoCaptureFile(streamURL)
	if err != nil {
		fmt.Printf("Error opening video capture streamURL: %v\n", streamURL)
		return
	}
	defer webcam.Close()

	var feedWindow *gocv.Window
	var blobWindow *gocv.Window

	if showWindowsFlag {
		fmt.Println(showWindowsFlag)
		feedWindow = gocv.NewWindow("Video Feed")
		defer feedWindow.Close()

		blobWindow = gocv.NewWindow("Blobs")
		defer blobWindow.Close()
	}

	img := gocv.NewMat()
	defer img.Close()

	imgDelta := gocv.NewMat()
	defer imgDelta.Close()

	imgThresh := gocv.NewMat()
	defer imgThresh.Close()

	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close()

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

		for _, id := range tracker.NewObjects {

			cars[id] = &Car{
				Track:   []CarTrack{},
				Tracker: contrib.NewTrackerMOSSE(),
			}

			defer cars[id].Tracker.Close()
			cars[id].Tracker.Init(img, tracker.Objects[id].CurrentRect)
		}

		for i, _ := range tracker.Objects {
			car := cars[i]

			if car == nil { //// TODO: Fix nil pointer dereference on missing tracker object
				continue
			}

			rect, _ := car.Tracker.Update(img)

			newPoint := image.Pt((rect.Min.X*2+rect.Dx())/2, (rect.Min.Y*2+rect.Dy())/2)

			gocv.Rectangle(&img, rect, color.RGBA{255, 0, 0, 0}, 1)

			for i := 0; i < len(car.Track)-2; i++ {
				gocv.Line(&img, car.Track[i].TrackPoint.Point, car.Track[i+1].TrackPoint.Point, color.RGBA{255, 0, 0, 0}, 1)
			}

			frameClone := img.Clone()
			frameClone = frameClone.Region(image.Rect(0, 0, 640, 190)) //Just show road in frame
			defer frameClone.Close()

			if newPoint.X > 0 && newPoint.Y > 0 {
				cars[i].Track = append(car.Track, CarTrack{
					TrackPoint: blob.NewTrackPoint(newPoint),
					Mat:        &frameClone,
				})
			}

		}

		if len(tracker.Objects) == 0 && len(cars) > 0 {
			for i, _ := range cars {
				removeCar(carMessageChan, cars, i)
			}

			cars = make(CarRegister)
			continue
		}

		carIDs := make([]uuid.UUID, 0, len(cars))
		for k := range cars {
			carIDs = append(carIDs, k)
		}

		for _, i := range carIDs {
			for o, _ := range tracker.Objects {
				if o == i {
					continue
				}

				removeCar(carMessageChan, cars, i)
			}
		}

		streamClone := img.Clone()
		streamClone = streamClone.Region(image.Rect(0, 0, 640, 190)) //Just show road in frame
		defer streamClone.Close()

		trackingStream.Channel <- streamClone

		if showWindowsFlag {
			feedWindow.IMShow(img)
			blobWindow.IMShow(imgThresh)
		}

		// if window.WaitKey(1) == 27 {
		// 	break
		// }
	}
}
