import serial
import time


def main():
    port = "/dev/ttyUSB0"   # Linux/Mac
    baud = 9600             # Match with Arduino Serial.begin(9600)

    try:
        # Open serial connection
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Wait for Arduino reset after connection
        print("Connected to Arduino on", port)

    except:
        raise ExceptionError

        # Send data to Arduino
        ser.write(b"Hello Arduino!\n")
        print("Sent: Hello Arduino!")

        # Read response
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").rstrip()
                print("Received:", line)

                # Example: exit if Arduino sends "bye"
                if line.lower() == "bye":
                    break

        ser.close()

    except serial.SerialException as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
