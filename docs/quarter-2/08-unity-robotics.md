---
title: "Chapter 8: Unity Robotics"
sidebar_label: "8. Unity Robotics"
sidebar_position: 8
---

import { PythonCode } from '@site/src/components/CodeBlock';
import { BashCode } from '@site/src/components/CodeBlock';
import { ROS2Code } from '@site/src/components/CodeBlock';

# Chapter 8: Unity Robotics

## Advanced 3D Simulation and Visualization for Robotics

Welcome to Chapter 8, where we explore Unity's powerful capabilities for robotics simulation, visualization, and training. Unity's advanced rendering engine, physics system, and extensive asset ecosystem make it an exceptional platform for creating photorealistic robot simulations, interactive training environments, and augmented reality robotics applications.

## 🎯 Chapter Learning Objectives

By the end of this chapter, you will be able to:

1. **Master Unity-ROS Integration**: Connect Unity simulations with ROS 2 for seamless data exchange
2. **Create Photorealistic Environments**: Design visually stunning simulation worlds with advanced lighting and materials
3. **Implement Robot Control Systems**: Build sophisticated robot controllers within Unity's architecture
4. **Develop Sensor Simulations**: Create realistic camera, LiDAR, and depth sensor implementations
5. **Build Training Applications**: Create interactive robotics training and simulation applications

## 🏗️ Unity Robotics Architecture

### Unity-ROS 2 Bridge

The core component connecting Unity with ROS 2:

<PythonCode title="Unity-ROS2 Bridge Architecture">
```csharp
// Unity C# script for ROS2 integration
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;
using RosMessageTypes.Nav;

public class UnityROS2Bridge : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public string robotName = "unity_robot";
    public float updateRate = 30.0f;

    private ROSConnection rosConnection;
    private float timeSinceLastUpdate = 0f;

    // Publishers
    private string odomTopic;
    private string jointStateTopic;
    private string imageTopic;

    // Subscribers
    private string cmdVelTopic;
    private string jointTrajectoryTopic;

    void Start()
    {
        InitializeROSConnection();
        SetupPublishers();
        SetupSubscribers();
    }

    void InitializeROSConnection()
    {
        // Connect to ROS2
        rosConnection = ROSConnection.instance;
        rosConnection.InitializeROS(rosIP, rosPort);

        // Generate topic names
        odomTopic = $"/{robotName}/odom";
        jointStateTopic = $"/{robotName}/joint_states";
        imageTopic = $"/{robotName}/camera/image_raw";
        cmdVelTopic = $"/{robotName}/cmd_vel";
        jointTrajectoryTopic = $"/{robotName}/joint_trajectory";
    }

    void SetupPublishers()
    {
        // Register publishers
        rosConnection.RegisterPublisher<OdometryMsg>(odomTopic);
        rosConnection.RegisterPublisher<JointStateMsg>(jointStateTopic);
        rosConnection.RegisterPublisher<ImageMsg>(imageTopic);
    }

    void SetupSubscribers()
    {
        // Register subscribers
        rosConnection.Subscribe<TwistMsg>(cmdVelTopic, ReceiveTwistCommand);
        rosConnection.Subscribe<JointTrajectoryMsg>(jointTrajectoryTopic, ReceiveTrajectoryCommand);
    }

    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;

        if (timeSinceLastUpdate >= 1.0f / updateRate)
        {
            PublishRobotState();
            timeSinceLastUpdate = 0f;
        }
    }

    void PublishRobotState()
    {
        // Publish odometry
        PublishOdometry();

        // Publish joint states
        PublishJointStates();

        // Publish camera images
        PublishCameraImage();
    }

    void ReceiveTwistCommand(TwistMsg msg)
    {
        // Process velocity command
        Vector3 linearVelocity = new Vector3((float)msg.linear.x, (float)msg.linear.y, (float)msg.linear.z);
        Vector3 angularVelocity = new Vector3((float)msg.angular.x, (float)msg.angular.y, (float)msg.angular.z);

        // Apply to robot
        ApplyVelocityCommand(linearVelocity, angularVelocity);
    }

    void ReceiveTrajectoryCommand(JointTrajectoryMsg msg)
    {
        // Process joint trajectory command
        foreach (var point in msg.points)
        {
            ProcessTrajectoryPoint(point);
        }
    }
}
```
</PythonCode>

### Robot Controller Components

<PythonCode title="Unity Robot Controller Base Class">
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public abstract class RobotController : MonoBehaviour
{
    [Header("Robot Properties")]
    public float maxLinearSpeed = 2.0f;
    public float maxAngularSpeed = 1.0f;
    public float wheelRadius = 0.1f;
    public float wheelBase = 0.4f;

    [Header("Sensors")]
    public CameraSensor cameraSensor;
    public LidarSensor lidarSensor;
    public IMUSensor imuSensor;

    protected ROSConnection rosConnection;
    protected Rigidbody rigidbody;
    protected Transform robotTransform;

    protected virtual void Awake()
    {
        rigidbody = GetComponent<Rigidbody>();
        robotTransform = transform;
        rosConnection = ROSConnection.instance;
    }

    protected virtual void Start()
    {
        InitializeSensors();
        SetupROSCommunication();
    }

    protected virtual void InitializeSensors()
    {
        if (cameraSensor != null)
            cameraSensor.Initialize(rosConnection);

        if (lidarSensor != null)
            lidarSensor.Initialize(rosConnection);

        if (imuSensor != null)
            imuSensor.Initialize(rosConnection);
    }

    public abstract void ApplyVelocityCommand(Vector3 linear, Vector3 angular);
    public abstract void SetJointPositions(float[] jointPositions);
    public abstract void PublishRobotState();
}

// Differential Drive Implementation
public class DifferentialDriveController : RobotController
{
    [Header("Wheels")]
    public Transform leftWheel;
    public Transform rightWheel;
    public float motorTorque = 10.0f;

    private WheelCollider leftWheelCollider;
    private WheelCollider rightWheelCollider;

    protected override void Awake()
    {
        base.Awake();

        // Get wheel colliders
        leftWheelCollider = leftWheel.GetComponent<WheelCollider>();
        rightWheelCollider = rightWheel.GetComponent<WheelCollider>();
    }

    public override void ApplyVelocityCommand(Vector3 linear, Vector3 angular)
    {
        // Calculate wheel velocities
        float v = linear.x;  // Forward velocity
        float omega = angular.z;  // Angular velocity

        float leftWheelVelocity = (v - omega * wheelBase / 2.0f) / wheelRadius;
        float rightWheelVelocity = (v + omega * wheelBase / 2.0f) / wheelRadius;

        // Apply motor torque
        leftWheelCollider.motorTorque = leftWheelVelocity * motorTorque;
        rightWheelCollider.motorTorque = rightWheelVelocity * motorTorque;
    }

    public override void SetJointPositions(float[] jointPositions)
    {
        // Not applicable for differential drive
    }

    public override void PublishRobotState()
    {
        // Publish odometry
        PublishOdometry();

        // Publish wheel velocities
        PublishWheelVelocities();
    }

    private void PublishOdometry()
    {
        var odomMsg = new OdometryMsg();

        // Header
        odomMsg.header.stamp = Time.time.ToRosTime();
        odomMsg.header.frame_id = "odom";
        odomMsg.child_frame_id = "base_link";

        // Position
        odomMsg.pose.pose.position.x = robotTransform.position.x;
        odomMsg.pose.pose.position.y = robotTransform.position.z;  // Unity Z -> ROS Y
        odomMsg.pose.pose.position.z = robotTransform.position.y;  // Unity Y -> ROS Z

        // Orientation (Unity to ROS coordinate conversion)
        Quaternion unityOrientation = robotTransform.rotation;
        Vector3 rosEuler = ConvertUnityToRosEuler(unityOrientation);
        odomMsg.pose.pose.orientation = rosEuler.ToRosQuaternion();

        // Twist
        Vector3 velocity = rigidbody.velocity;
        odomMsg.twist.twist.linear.x = velocity.x;
        odomMsg.twist.twist.linear.y = velocity.z;
        odomMsg.twist.twist.angular.z = rigidbody.angularVelocity.y;

        // Publish
        rosConnection.Publish($"/{gameObject.name}/odom", odomMsg);
    }

    private Vector3 ConvertUnityToRosEuler(Quaternion unityQuat)
    {
        // Convert Unity coordinate system to ROS coordinate system
        Vector3 unityEuler = unityQuat.eulerAngles;
        return new Vector3(
            unityEuler.z * Mathf.Deg2Rad,  // Unity Z -> ROS X
            unityEuler.y * Mathf.Deg2Rad,  // Unity Y -> ROS Y
            -unityEuler.x * Mathf.Deg2Rad  // Unity X -> -ROS Z
        );
    }
}
```
</PythonCode>

## 📡 Sensor Simulation in Unity

### Camera Sensor Implementation

<PythonCode title="High-Fidelity Camera Sensor">
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;
using System.Runtime.InteropServices;

public class CameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int imageWidth = 1920;
    public int imageHeight = 1080;
    public float fieldOfView = 60.0f;
    public float nearClipPlane = 0.1f;
    public float farClipPlane = 100.0f;

    [Header("Camera Effects")]
    public bool enableDepth = true;
    public bool enableLensingDistortion = true;
    public float distortionK1 = 0.1f;
    public float distortionK2 = -0.05f;

    [Header("Publishing")]
    public string imageTopic = "/camera/image_raw";
    public string cameraInfoTopic = "/camera/camera_info";
    public float publishRate = 30.0f;

    private Camera cameraComponent;
    private RenderTexture renderTexture;
    private Texture2D outputTexture;
    private byte[] imageData;

    private ROSConnection rosConnection;
    private float timeSinceLastPublish = 0f;

    void Start()
    {
        SetupCamera();
        InitializeRenderTexture();
        rosConnection = ROSConnection.instance;
        rosConnection.RegisterPublisher<ImageMsg>(imageTopic);
        rosConnection.RegisterPublisher<CameraInfoMsg>(cameraInfoTopic);
    }

    void SetupCamera()
    {
        cameraComponent = GetComponent<Camera>();
        cameraComponent.fieldOfView = fieldOfView;
        cameraComponent.nearClipPlane = nearClipPlane;
        cameraComponent.farClipPlane = farClipPlane;
    }

    void InitializeRenderTexture()
    {
        // Create render texture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        renderTexture.Create();
        cameraComponent.targetTexture = renderTexture;

        // Create output texture
        outputTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        imageData = new byte[imageWidth * imageHeight * 3];
    }

    void Update()
    {
        timeSinceLastPublish += Time.deltaTime;

        if (timeSinceLastPublish >= 1.0f / publishRate)
        {
            CaptureAndPublishImage();
            timeSinceLastPublish = 0f;
        }
    }

    void CaptureAndPublishImage()
    {
        // Render current frame
        cameraComponent.Render();

        // Copy render texture to texture2D
        RenderTexture.active = renderTexture;
        outputTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        outputTexture.Apply();

        // Apply lens distortion if enabled
        if (enableLensingDistortion)
        {
            ApplyLensingDistortion();
        }

        // Convert to ROS Image message
        ImageMsg imageMsg = CreateImageMessage();

        // Publish image
        rosConnection.Publish(imageTopic, imageMsg);

        // Publish camera info
        CameraInfoMsg cameraInfoMsg = CreateCameraInfoMessage();
        rosConnection.Publish(cameraInfoTopic, cameraInfoMsg);
    }

    void ApplyLensingDistortion()
    {
        // Implement barrel/pincushion distortion
        Color[] pixels = outputTexture.GetPixels();
        Color[] distortedPixels = new Color[pixels.Length];

        float centerX = imageWidth / 2.0f;
        float centerY = imageHeight / 2.0f;
        float maxRadius = Mathf.Sqrt(centerX * centerX + centerY * centerY);

        for (int y = 0; y < imageHeight; y++)
        {
            for (int x = 0; x < imageWidth; x++)
            {
                // Convert to normalized coordinates
                float dx = (x - centerX) / maxRadius;
                float dy = (y - centerY) / maxRadius;
                float r = Mathf.Sqrt(dx * dx + dy * dy);

                // Apply distortion
                float distortionFactor = 1.0f + distortionK1 * r * r + distortionK2 * r * r * r * r;

                // Map back to pixel coordinates
                float srcX = centerX + dx * distortionFactor * maxRadius;
                float srcY = centerY + dy * distortionFactor * maxRadius;

                // Bilinear interpolation
                distortedPixels[y * imageWidth + x] = BilinearInterpolation(pixels, srcX, srcY, imageWidth, imageHeight);
            }
        }

        outputTexture.SetPixels(distortedPixels);
    }

    Color BilinearInterpolation(Color[] pixels, float x, float y, int width, int height)
    {
        int x1 = Mathf.FloorToInt(x);
        int y1 = Mathf.FloorToInt(y);
        int x2 = Mathf.Min(x1 + 1, width - 1);
        int y2 = Mathf.Min(y1 + 1, height - 1);

        float fx = x - x1;
        float fy = y - y1;

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
        {
            Color c11 = pixels[y1 * width + x1];
            Color c21 = pixels[y1 * width + x2];
            Color c12 = pixels[y2 * width + x1];
            Color c22 = pixels[y2 * width + x2];

            return Color.Lerp(
                Color.Lerp(c11, c21, fx),
                Color.Lerp(c12, c22, fx),
                fy
            );
        }

        return Color.black;
    }

    ImageMsg CreateImageMessage()
    {
        ImageMsg imageMsg = new ImageMsg();

        // Header
        imageMsg.header.stamp = Time.time.ToRosTime();
        imageMsg.header.frame_id = "camera_optical_frame";

        // Image properties
        imageMsg.height = imageHeight;
        imageMsg.width = imageWidth;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = imageWidth * 3;

        // Convert texture to byte array
        imageData = outputTexture.GetRawTextureData();
        imageMsg.data = imageData;

        return imageMsg;
    }

    CameraInfoMsg CreateCameraInfoMessage()
    {
        CameraInfoMsg cameraInfo = new CameraInfoMsg();

        // Header
        cameraInfo.header.stamp = Time.time.ToRosTime();
        cameraInfo.header.frame_id = "camera_optical_frame";

        // Image dimensions
        cameraInfo.height = imageHeight;
        cameraInfo.width = imageWidth;

        // Distortion model
        cameraInfo.distortion_model = "plumb_bob";
        cameraInfo.D = new double[] { distortionK1, distortionK2, 0.0, 0.0, 0.0 };

        // Camera matrix (intrinsic parameters)
        float fx = imageWidth / (2.0f * Mathf.Tan(fieldOfView * 0.5f * Mathf.Deg2Rad));
        float fy = fx;
        float cx = imageWidth / 2.0f;
        float cy = imageHeight / 2.0f;

        cameraInfo.K = new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 };

        // Rectification matrix
        cameraInfo.R = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

        // Projection matrix
        cameraInfo.P = new double[] { fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 };

        // Binning and ROI
        cameraInfo.binning_x = 1;
        cameraInfo.binning_y = 1;
        cameraInfo.roi.x_offset = 0;
        cameraInfo.roi.y_offset = 0;
        cameraInfo.roi.height = imageHeight;
        cameraInfo.roi.width = imageWidth;
        cameraInfo.roi.do_rectify = false;

        return cameraInfo;
    }
}
```
</PythonCode>

### LiDAR Sensor Simulation

<PythonCode title="Realistic LiDAR Sensor Implementation">
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections;
using System.Collections.Generic;

public class LidarSensor : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public int horizontalSamples = 360;
    public int verticalLayers = 16;
    public float maxRange = 100.0f;
    public float minRange = 0.1f;
    public float frequency = 10.0f;

    [Header("Beam Properties")]
    public float beamDivergence = 0.25f;
    public bool enableNoise = true;
    public float noiseStdDev = 0.01f;

    [Header("Scan Pattern")]
    public float horizontalFOV = 360.0f;
    public float verticalFOV = 30.0f;
    public bool enableVerticalScanning = true;

    private string scanTopic = "/laser/scan";
    private string pointCloudTopic = "/laser/pointcloud";

    private ROSConnection rosConnection;
    private LayerMask obstacleLayer;

    void Start()
    {
        rosConnection = ROSConnection.instance;
        rosConnection.RegisterPublisher<LaserScanMsg>(scanTopic);
        rosConnection.RegisterPublisher<PointCloud2Msg>(pointCloudTopic);

        obstacleLayer = LayerMask.GetMask("Default", "Obstacle");
    }

    public void Initialize(ROSConnection connection)
    {
        rosConnection = connection;
    }

    public void PerformScan()
    {
        StartCoroutine(ScanRoutine());
    }

    private IEnumerator ScanRoutine()
    {
        float[] ranges = new float[horizontalSamples];
        float[] intensities = new float[horizontalSamples];

        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = (i / (float)horizontalSamples) * horizontalFOV - horizontalFOV / 2.0f;
            Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;

            // Perform raycast
            RaycastHit hit;
            bool hasHit = Physics.Raycast(transform.position, direction, out hit, maxRange, obstacleLayer);

            if (hasHit)
            {
                ranges[i] = hit.distance;
                intensities[i] = CalculateIntensity(hit);
            }
            else
            {
                ranges[i] = maxRange;
                intensities[i] = 0.0f;
            }

            // Add noise if enabled
            if (enableNoise)
            {
                ranges[i] += GaussianRandom(0.0f, noiseStdDev);
            }

            yield return null;
        }

        // Publish laser scan
        PublishLaserScan(ranges, intensities);

        // Generate and publish point cloud
        if (enableVerticalScanning)
        {
            yield return StartCoroutine(GeneratePointCloud());
        }
    }

    private float CalculateIntensity(RaycastHit hit)
    {
        // Calculate intensity based on material properties and angle
        Vector3 incident = (transform.position - hit.point).normalized;
        Vector3 normal = hit.normal;
        float angle = Vector3.Angle(incident, normal);

        // Material-dependent intensity
        float materialIntensity = 1.0f;
        if (hit.collider.gameObject.CompareTag("Metal"))
            materialIntensity = 0.9f;
        else if (hit.collider.gameObject.CompareTag("Plastic"))
            materialIntensity = 0.6f;
        else if (hit.collider.gameObject.CompareTag("Glass"))
            materialIntensity = 0.3f;

        return materialIntensity * Mathf.Cos(angle * Mathf.Deg2Rad);
    }

    private IEnumerator GeneratePointCloud()
    {
        List<Vector3> points = new List<Vector3>();
        List<Color> colors = new List<Color>();

        for (int h = 0; h < horizontalSamples; h += 2)  // Sample every 2nd horizontal ray for performance
        {
            for (int v = 0; v < verticalLayers; v++)
            {
                float hAngle = (h / (float)horizontalSamples) * horizontalFOV - horizontalFOV / 2.0f;
                float vAngle = (v / (float)(verticalLayers - 1)) * verticalFOV - verticalFOV / 2.0f;

                Quaternion rotation = Quaternion.Euler(-vAngle, hAngle, 0);
                Vector3 direction = rotation * Vector3.forward;

                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, maxRange, obstacleLayer))
                {
                    points.Add(transform.InverseTransformPoint(hit.point));
                    colors.Add(new Color(hit.distance / maxRange, 0, 1 - hit.distance / maxRange, 1));
                }
            }

            yield return null;  // Spread over multiple frames
        }

        PublishPointCloud(points, colors);
    }

    private void PublishLaserScan(float[] ranges, float[] intensities)
    {
        LaserScanMsg scanMsg = new LaserScanMsg();

        // Header
        scanMsg.header.stamp = Time.time.ToRosTime();
        scanMsg.header.frame_id = "laser_frame";

        // Scan parameters
        scanMsg.angle_min = -horizontalFOV / 2.0f * Mathf.Deg2Rad;
        scanMsg.angle_max = horizontalFOV / 2.0f * Mathf.Deg2Rad;
        scanMsg.angle_increment = (horizontalFOV / horizontalSamples) * Mathf.Deg2Rad;
        scanMsg.time_increment = 0.0f;
        scanMsg.scan_time = 1.0f / frequency;

        // Range parameters
        scanMsg.range_min = minRange;
        scanMsg.range_max = maxRange;

        // Data
        scanMsg.ranges = ranges;
        scanMsg.intensities = intensities;

        // Publish
        rosConnection.Publish(scanTopic, scanMsg);
    }

    private void PublishPointCloud(List<Vector3> points, List<Color> colors)
    {
        PointCloud2Msg pointCloudMsg = new PointCloud2Msg();

        // Header
        pointCloudMsg.header.stamp = Time.time.ToRosTime();
        pointCloudMsg.header.frame_id = "laser_frame";

        // Point cloud properties
        pointCloudMsg.height = 1;
        pointCloudMsg.width = points.Count;
        pointCloudMsg.is_bigendian = false;
        pointCloudMsg.point_step = 16;  // x(4) + y(4) + z(4) + rgb(4)
        pointCloudMsg.row_step = pointCloudMsg.width * pointCloudMsg.point_step;

        // Fields
        pointCloudMsg.fields = new PointFieldMsg[]
        {
            new PointFieldMsg { name = "x", offset = 0, datatype = 7, count = 1 },  // Float32
            new PointFieldMsg { name = "y", offset = 4, datatype = 7, count = 1 },
            new PointFieldMsg { name = "z", offset = 8, datatype = 7, count = 1 },
            new PointFieldMsg { name = "rgb", offset = 12, datatype = 7, count = 1 }
        };

        // Data
        byte[] data = new byte[points.Count * pointCloudMsg.point_step];
        int dataIndex = 0;

        for (int i = 0; i < points.Count; i++)
        {
            // Convert Unity coordinates to ROS coordinates
            Vector3 rosPoint = ConvertUnityToRos(points[i]);
            Color color = colors[i];

            // Pack float values into bytes
            byte[] xBytes = System.BitConverter.GetBytes(rosPoint.x);
            byte[] yBytes = System.BitConverter.GetBytes(rosPoint.y);
            byte[] zBytes = System.BitConverter.GetBytes(rosPoint.z);

            // Pack color into integer
            int rgbInt = ((int)(color.r * 255) << 16) | ((int)(color.g * 255) << 8) | ((int)(color.b * 255));
            byte[] rgbBytes = System.BitConverter.GetBytes(rgbInt);

            // Copy to data array
            System.Buffer.BlockCopy(xBytes, 0, data, dataIndex, 4);
            System.Buffer.BlockCopy(yBytes, 0, data, dataIndex + 4, 4);
            System.Buffer.BlockCopy(zBytes, 0, data, dataIndex + 8, 4);
            System.Buffer.BlockCopy(rgbBytes, 0, data, dataIndex + 12, 4);

            dataIndex += pointCloudMsg.point_step;
        }

        pointCloudMsg.data = data;

        // Publish
        rosConnection.Publish(pointCloudTopic, pointCloudMsg);
    }

    private Vector3 ConvertUnityToRos(Vector3 unityPoint)
    {
        return new Vector3(unityPoint.x, unityPoint.z, unityPoint.y);
    }

    private float GaussianRandom(float mean, float stdDev)
    {
        float u1 = Random.Range(0.0f, 1.0f);
        float u2 = Random.Range(0.0f, 1.0f);
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);

        return mean + stdDev * randStdNormal;
    }
}
```
</PythonCode>

## 🎨 Advanced Rendering and Materials

### Physically Based Rendering (PBR) for Robotics

<PythonCode title="PBR Material Setup for Robot Components">
```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "RobotMaterialPBR", menuName = "Robotics/RobotMaterialPBR")]
public class RobotMaterialPBR : ScriptableObject
{
    [Header("Material Properties")]
    public MaterialType materialType = MaterialType.Metal;

    [Header("Base Properties")]
    [Range(0, 1)] public float metallic = 0.0f;
    [Range(0, 1)] public float smoothness = 0.5f;
    [ColorUsage(false, true)] public Color baseColor = Color.white;

    [Header("Normal Mapping")]
    public Texture2D normalMap;
    [Range(0, 2)] public float normalStrength = 1.0f;

    [Header("Surface Detail")]
    [Range(0, 1)] public float occlusionStrength = 1.0f;
    [Range(0, 10)] public float emissionStrength = 0.0f;
    public Color emissionColor = Color.black;

    [Header("Environmental Response")]
    [Range(0, 1)] public float reflectance = 0.5f;
    [Range(0, 2)] public float anisotropy = 0.0f;

    public enum MaterialType
    {
        Metal,
        Plastic,
        Rubber,
        Glass,
        Ceramic,
        Composite
    }

    public Material CreateMaterial()
    {
        Material material = new Material(Shader.Find("Universal Render Pipeline/Lit"));

        ApplyMaterialType(material);

        // Set base properties
        material.SetColor("_BaseColor", baseColor);
        material.SetFloat("_Metallic", metallic);
        material.SetFloat("_Smoothness", smoothness);

        // Set normal mapping
        if (normalMap != null)
        {
            material.SetTexture("_BaseColorMap", normalMap);
            material.SetFloat("_NormalScale", normalStrength);
        }

        // Set emission
        if (emissionStrength > 0)
        {
            material.EnableKeyword("_EMISSION");
            material.SetColor("_EmissionColor", emissionColor * emissionStrength);
        }

        // Set environmental response
        material.SetFloat("_SurfaceType", metallic > 0.5f ? 1 : 0);  // 0 = opaque, 1 = transparent
        material.SetFloat("_BlendMode", 0);  // 0 = alpha, 1 = premultiply, 2 = additive

        return material;
    }

    private void ApplyMaterialType(Material material)
    {
        switch (materialType)
        {
            case MaterialType.Metal:
                metallic = 1.0f;
                smoothness = 0.8f;
                reflectance = 0.9f;
                break;

            case MaterialType.Plastic:
                metallic = 0.0f;
                smoothness = 0.6f;
                reflectance = 0.4f;
                break;

            case MaterialType.Rubber:
                metallic = 0.0f;
                smoothness = 0.2f;
                reflectance = 0.1f;
                break;

            case MaterialType.Glass:
                metallic = 0.0f;
                smoothness = 1.0f;
                reflectance = 0.8f;
                material.SetFloat("_SurfaceType", 1);  // Transparent
                break;

            case MaterialType.Ceramic:
                metallic = 0.0f;
                smoothness = 0.9f;
                reflectance = 0.7f;
                break;

            case MaterialType.Composite:
                metallic = 0.2f;
                smoothness = 0.5f;
                reflectance = 0.5f;
                break;
        }
    }
}
```
</PythonCode>

### Advanced Lighting Setup

<PythonCode title="Dynamic Lighting System for Robotics Simulations">
```csharp
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class RobotSimulationLighting : MonoBehaviour
{
    [Header("Scene Lighting")]
    public Light sunLight;
    public Color sunColor = Color.white;
    [Range(0, 3)] public float sunIntensity = 1.0f;

    [Header("Environment Lighting")]
    public Material skyboxMaterial;
    [ColorUsage(false, true)] public Color ambientColor = new Color(0.2f, 0.3f, 0.4f, 1.0f);
    [Range(0, 2)] public float ambientIntensity = 1.0f;

    [Header("Real-time Lighting")]
    public bool enableRealtimeGI = true;
    public int realtimeBounces = 2;
    public float realtimeResolution = 1.0f;

    [Header("Shadow Settings")]
    public bool enableShadows = true;
    public int shadowCascadeCount = 4;
    [Range(0.001f, 1f)] public float shadowDistance = 150.0f;

    [Header("Post Processing")]
    public Volume postProcessVolume;
    public bool enableBloom = true;
    public bool enableVignette = true;
    public bool enableColorGrading = true;

    private UniversalAdditionalCameraData cameraData;
    private VolumeProfile volumeProfile;

    void Start()
    {
        SetupLighting();
        SetupPostProcessing();
        ConfigureRenderingSettings();
    }

    void SetupLighting()
    {
        // Configure directional light (sun)
        if (sunLight != null)
        {
            sunLight.type = LightType.Directional;
            sunLight.color = sunColor;
            sunLight.intensity = sunIntensity;
            sunLight.shadows = enableShadows ? LightShadows.Soft : LightShadows.None;

            // Set sun position based on time of day
            UpdateSunPosition(6.0f);  // 6 AM
        }

        // Configure ambient lighting
        RenderSettings.ambientMode = AmbientMode.Skybox;
        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientSkybox = skyboxMaterial;

        // Configure reflection probe
        SetupReflectionProbe();
    }

    void SetupReflectionProbe()
    {
        GameObject reflectionProbeGO = new GameObject("RobotReflectionProbe");
        ReflectionProbe reflectionProbe = reflectionProbeGO.AddComponent<ReflectionProbe>();

        // Configure probe
        reflectionProbe.size = new Vector3(50, 20, 50);
        reflectionProbe.center = Vector3.zero;
        reflectionProbe.intensity = 1.0f;
        reflectionProbe.resolution = 256;
        reflectionProbe.clearFlags = ReflectionProbeClearFlags.Skybox;

        // Set importance for proper blending
        reflectionProbe.importance = 1.0f;

        // Set culling mask for robot components
        reflectionProbe.cullingMask = LayerMask.GetMask("Robot", "Environment");

        // Bake the probe
        reflectionProbe.RenderProbe();
    }

    void SetupPostProcessing()
    {
        if (postProcessVolume == null)
        {
            GameObject volumeGO = new GameObject("PostProcessVolume");
            volumeGO.AddComponent<Volume>();
            postProcessVolume = volumeGO.GetComponent<Volume>();
        }

        // Create volume profile
        volumeProfile = ScriptableObject.CreateInstance<VolumeProfile>();
        postProcessVolume.profile = volumeProfile;
        postProcessVolume.isGlobal = true;

        if (enableBloom)
        {
            AddBloomEffect();
        }

        if (enableVignette)
        {
            AddVignetteEffect();
        }

        if (enableColorGrading)
        {
            AddColorGradingEffect();
        }
    }

    void AddBloomEffect()
    {
        Bloom bloom = volumeProfile.Add<Bloom>(true);
        bloom.threshold.value = 1.0f;
        bloom.intensity.value = 0.5f;
        bloom.color.value = Color.white;
        bloom.fastMode.value = true;
    }

    void AddVignetteEffect()
    {
        Vignette vignette = volumeProfile.Add<Vignette>(true);
        vignette.color.value = Color.black;
        vignette.center.value = new Vector2(0.5f, 0.5f);
        vignette.intensity.value = 0.3f;
        vignette.roundness.value = 1.0f;
        vignette.smoothness.value = 0.5f;
    }

    void AddColorGradingEffect()
    {
        ColorGrading colorGrading = volumeProfile.Add<ColorGrading>(true);
        colorGrading.contrast.value = 10.0f;
        colorGrading.hueShift.value = 0.0f;
        colorGrading.saturation.value = 10.0f;
        colorGrading.temperature.value = 0.0f;
        colorGrading.tint.value = 0.0f;
    }

    void ConfigureRenderingSettings()
    {
        // Get URP asset
        UniversalRenderPipelineAsset urpAsset = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;

        if (urpAsset != null)
        {
            // Configure shadow settings
            urpAsset.shadowDistance = shadowDistance;
            urpAsset.shadowCascadeCount = shadowCascadeCount;

            // Configure quality settings
            urpAsset.renderScale = 1.0f;
            urpAsset.hdr = true;

            // Configure lighting settings
            urpAsset.mainLightRenderingMode = LightRenderingMode.PerPixel;
            urpAsset.additionalLightsRenderingMode = LightRenderingMode.PerPixel;
            urpAsset.additionalLightsPerObjectLimit = 4;
        }
    }

    void UpdateSunPosition(float timeOfDay)
    {
        // Calculate sun angle based on time of day (0-24 hours)
        float sunAngle = (timeOfDay / 24.0f) * 360.0f - 90.0f;

        Quaternion sunRotation = Quaternion.Euler(sunAngle, 30.0f, 0.0f);
        sunLight.transform.rotation = sunRotation;

        // Adjust sun color based on time
        if (timeOfDay >= 6.0f && timeOfDay <= 18.0f)  // Daytime
        {
            sunLight.color = Color.white;
            sunLight.intensity = 1.0f;
        }
        else  // Nighttime
        {
            sunLight.color = new Color(0.2f, 0.3f, 0.5f);  // Moonlight blue
            sunLight.intensity = 0.1f;
        }
    }

    public void EnableRealtimeLighting(bool enable)
    {
        enableRealtimeGI = enable;

        DynamicGI.UpdateEnvironment();

        if (enable)
        {
            DynamicGI.UpdateMaterials();
        }
    }

    public void UpdateLightingForEnvironment(EnvironmentType environment)
    {
        switch (environment)
        {
            case EnvironmentType.Indoor:
                ConfigureIndoorLighting();
                break;

            case EnvironmentType.Outdoor:
                ConfigureOutdoorLighting();
                break;

            case EnvironmentType.Underground:
                ConfigureUndergroundLighting();
                break;

            case EnvironmentType.Space:
                ConfigureSpaceLighting();
                break;
        }
    }

    void ConfigureIndoorLighting()
    {
        sunLight.intensity = 0.2f;
        ambientIntensity = 0.8f;
        ambientColor = new Color(0.4f, 0.4f, 0.5f, 1.0f);

        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;
    }

    void ConfigureOutdoorLighting()
    {
        sunLight.intensity = 1.0f;
        ambientIntensity = 0.3f;
        ambientColor = new Color(0.2f, 0.3f, 0.4f, 1.0f);

        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;
    }

    void ConfigureUndergroundLighting()
    {
        sunLight.intensity = 0.0f;
        ambientIntensity = 0.1f;
        ambientColor = new Color(0.1f, 0.1f, 0.2f, 1.0f);

        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;

        // Add spotlights for artificial lighting
        AddSpotlights();
    }

    void ConfigureSpaceLighting()
    {
        sunLight.intensity = 1.5f;
        ambientIntensity = 0.0f;
        ambientColor = Color.black;

        RenderSettings.ambientIntensity = ambientIntensity;
        RenderSettings.ambientLight = ambientColor;

        // Configure starfield skybox
        RenderSettings.skybox = skyboxMaterial;
    }

    void AddSpotlights()
    {
        GameObject lightGroup = new GameObject("ArtificialLighting");

        // Add multiple spotlights for underground environment
        for (int i = 0; i < 10; i++)
        {
            GameObject lightGO = new GameObject($"Spotlight_{i}");
            lightGO.transform.SetParent(lightGroup.transform);

            Light spotLight = lightGO.AddComponent<Light>();
            spotLight.type = LightType.Spot;
            spotLight.intensity = 2.0f;
            spotLight.range = 20.0f;
            spotLight.spotAngle = 45.0f;
            spotLight.color = new Color(1.0f, 0.8f, 0.6f);  // Warm artificial light

            // Position lights throughout the environment
            lightGO.transform.position = new Vector3(
                Random.Range(-20, 20),
                10.0f,
                Random.Range(-20, 20)
            );

            lightGO.transform.rotation = Quaternion.Euler(90, 0, 0);
        }
    }

    public enum EnvironmentType
    {
        Indoor,
        Outdoor,
        Underground,
        Space
    }
}
```
</PythonCode>

## 🎯 Interactive Training Applications

### Robot Operation Trainer

<PythonCode title="Interactive Robot Training Interface">
```csharp
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System.Collections;
using TMPro;

public class RobotOperationTrainer : MonoBehaviour
{
    [Header("Training Modules")]
    public TrainingModule[] trainingModules;
    public int currentModuleIndex = 0;

    [Header("UI Components")]
    public TextMeshProUGUI moduleTitleText;
    public TextMeshProUGUI instructionText;
    public TextMeshProUGUI scoreText;
    public TextMeshProUGUI timerText;
    public Slider progressBar;

    [Header("Feedback System")]
    public GameObject feedbackPanel;
    public TextMeshProUGUI feedbackText;
    public GameObject successEffect;
    public GameObject failureEffect;

    [Header("Performance Tracking")]
    public float[] moduleScores;
    public float[] completionTimes;
    public int attempts = 0;

    private TrainingModule currentModule;
    private float moduleStartTime;
    private bool isTrainingActive = false;

    void Start()
    {
        InitializeTraining();
    }

    void InitializeTraining()
    {
        moduleScores = new float[trainingModules.Length];
        completionTimes = new float[trainingModules.Length];

        LoadModule(0);
    }

    void Update()
    {
        if (isTrainingActive)
        {
            UpdateTimer();
            CheckModuleCompletion();
        }
    }

    public void LoadModule(int moduleIndex)
    {
        if (moduleIndex >= 0 && moduleIndex < trainingModules.Length)
        {
            currentModuleIndex = moduleIndex;
            currentModule = trainingModules[moduleIndex];

            // Update UI
            moduleTitleText.text = currentModule.moduleName;
            instructionText.text = currentModule.instructions;
            progressBar.value = (float)moduleIndex / (trainingModules.Length - 1);

            // Start module
            StartModule();
        }
    }

    void StartModule()
    {
        // Reset module state
        currentModule.Reset();
        moduleStartTime = Time.time;
        isTrainingActive = true;
        attempts = 0;

        // Hide feedback
        feedbackPanel.SetActive(false);

        // Enable module-specific functionality
        currentModule.StartModule(this);
    }

    void UpdateTimer()
    {
        float elapsedTime = Time.time - moduleStartTime;
        timerText.text = FormatTime(elapsedTime);
    }

    string FormatTime(float time)
    {
        int minutes = Mathf.FloorToInt(time / 60);
        int seconds = Mathf.FloorToInt(time % 60);
        return $"{minutes:00}:{seconds:00}";
    }

    void CheckModuleCompletion()
    {
        if (currentModule.CheckCompletion())
        {
            CompleteModule();
        }
    }

    void CompleteModule()
    {
        isTrainingActive = false;

        float completionTime = Time.time - moduleStartTime;
        completionTimes[currentModuleIndex] = completionTime;

        // Calculate score based on time and accuracy
        float score = currentModule.CalculateScore(completionTime, attempts);
        moduleScores[currentModuleIndex] = score;

        // Update score display
        UpdateOverallScore();

        // Show feedback
        ShowFeedback(true, score);

        // Play success effect
        if (successEffect != null)
        {
            successEffect.SetActive(true);
        }

        // Auto-advance to next module after delay
        StartCoroutine(AutoAdvanceModule());
    }

    void ShowFeedback(bool success, float score)
    {
        feedbackPanel.SetActive(true);

        if (success)
        {
            feedbackText.text = $"Excellent! Module completed!\nScore: {score:F1}%\nTime: {FormatTime(completionTimes[currentModuleIndex])}";
        }
        else
        {
            feedbackText.text = "Module failed. Try again!";
        }
    }

    void UpdateOverallScore()
    {
        float totalScore = 0;
        for (int i = 0; i < moduleScores.Length; i++)
        {
            totalScore += moduleScores[i];
        }

        float averageScore = totalScore / moduleScores.Length;
        scoreText.text = $"Score: {averageScore:F1}%";
    }

    IEnumerator AutoAdvanceModule()
    {
        yield return new WaitForSeconds(3.0f);

        if (successEffect != null)
        {
            successEffect.SetActive(false);
        }

        // Load next module
        int nextModule = currentModuleIndex + 1;
        if (nextModule < trainingModules.Length)
        {
            LoadModule(nextModule);
        }
        else
        {
            // Training completed
            CompleteTraining();
        }
    }

    void CompleteTraining()
    {
        // Show final results
        float totalScore = 0;
        float totalTime = 0;

        for (int i = 0; i < moduleScores.Length; i++)
        {
            totalScore += moduleScores[i];
            totalTime += completionTimes[i];
        }

        float averageScore = totalScore / moduleScores.Length;

        feedbackText.text = $"Training Complete!\nFinal Score: {averageScore:F1}%\nTotal Time: {FormatTime(totalTime)}\nModules Passed: {CountPassedModules()}/{trainingModules.Length}";

        // Save training data
        SaveTrainingData();
    }

    int CountPassedModules()
    {
        int count = 0;
        for (int i = 0; i < moduleScores.Length; i++)
        {
            if (moduleScores[i] >= 70.0f)  // Passing threshold
            {
                count++;
            }
        }
        return count;
    }

    void SaveTrainingData()
    {
        TrainingData data = new TrainingData
        {
            timestamp = System.DateTime.Now.ToString(),
            moduleName = "Robot Operation Training",
            moduleScores = moduleScores,
            completionTimes = completionTimes,
            totalScore = CalculateOverallScore(),
            passed = CountPassedModules() >= trainingModules.Length * 0.8f
        };

        // Save to file or database
        string jsonData = JsonUtility.ToJson(data, true);
        PlayerPrefs.SetString("LastTrainingData", jsonData);
        PlayerPrefs.Save();
    }

    public void OnModuleFailed()
    {
        attempts++;

        // Show failure feedback
        ShowFeedback(false, 0);

        // Play failure effect
        if (failureEffect != null)
        {
            failureEffect.SetActive(true);
        }

        // Restart module after delay
        StartCoroutine(RestartModule());
    }

    IEnumerator RestartModule()
    {
        yield return new WaitForSeconds(2.0f);

        if (failureEffect != null)
        {
            failureEffect.SetActive(false);
        }

        LoadModule(currentModuleIndex);
    }

    public void SkipModule()
    {
        if (currentModuleIndex < trainingModules.Length - 1)
        {
            LoadModule(currentModuleIndex + 1);
        }
    }

    public void RestartTraining()
    {
        InitializeTraining();
    }

    float CalculateOverallScore()
    {
        float total = 0;
        foreach (float score in moduleScores)
        {
            total += score;
        }
        return total / moduleScores.Length;
    }
}

[System.Serializable]
public class TrainingModule
{
    public string moduleName;
    [TextArea(3, 5)] public string instructions;
    public ModuleType moduleType;
    public float maxScore = 100.0f;
    public float timeBonus = 10.0f;
    public float targetTime = 60.0f;

    [Header("Module-specific Settings")]
    public float requiredAccuracy = 0.8f;
    public int maxAttempts = 3;

    private float currentAccuracy = 0.0f;
    private int currentAttempts = 0;

    public enum ModuleType
    {
        Navigation,
        Manipulation,
        Inspection,
        EmergencyResponse,
        Maintenance
    }

    public void StartModule(RobotOperationTrainer trainer)
    {
        currentAccuracy = 0.0f;
        currentAttempts = 0;

        switch (moduleType)
        {
            case ModuleType.Navigation:
                StartNavigationModule(trainer);
                break;

            case ModuleType.Manipulation:
                StartManipulationModule(trainer);
                break;

            case ModuleType.Inspection:
                StartInspectionModule(trainer);
                break;

            case ModuleType.EmergencyResponse:
                StartEmergencyModule(trainer);
                break;

            case ModuleType.Maintenance:
                StartMaintenanceModule(trainer);
                break;
        }
    }

    public void Reset()
    {
        currentAccuracy = 0.0f;
        currentAttempts = 0;

        // Reset module-specific state
        // This would be implemented per module type
    }

    public bool CheckCompletion()
    {
        return currentAccuracy >= requiredAccuracy;
    }

    public float CalculateScore(float completionTime, int attempts)
    {
        float baseScore = currentAccuracy * maxScore;

        // Time bonus
        float timeBonus = Mathf.Max(0, (targetTime - completionTime) / targetTime * this.timeBonus);

        // Attempt penalty
        float attemptPenalty = attempts * 5.0f;

        return Mathf.Max(0, baseScore + timeBonus - attemptPenalty);
    }

    private void StartNavigationModule(RobotOperationTrainer trainer)
    {
        // Implementation for navigation training
        Debug.Log("Starting Navigation Module");

        // Set up waypoints, obstacles, and targets
        // Track robot movement and collision avoidance
    }

    private void StartManipulationModule(RobotOperationTrainer trainer)
    {
        // Implementation for manipulation training
        Debug.Log("Starting Manipulation Module");

        // Set up objects to manipulate
        // Track gripper control and object placement
    }

    private void StartInspectionModule(RobotOperationTrainer trainer)
    {
        // Implementation for inspection training
        Debug.Log("Starting Inspection Module");

        // Set up inspection targets and defects
        // Track sensor data and defect identification
    }

    private void StartEmergencyModule(RobotOperationTrainer trainer)
    {
        // Implementation for emergency response training
        Debug.Log("Starting Emergency Response Module");

        // Set up emergency scenarios
        // Track response time and correct procedures
    }

    private void StartMaintenanceModule(RobotOperationTrainer trainer)
    {
        // Implementation for maintenance training
        Debug.Log("Starting Maintenance Module");

        // Set up maintenance tasks
        // Track procedure compliance and task completion
    }
}

[System.Serializable]
public class TrainingData
{
    public string timestamp;
    public string moduleName;
    public float[] moduleScores;
    public float[] completionTimes;
    public float totalScore;
    public bool passed;
}
```
</PythonCode>

## 📋 Chapter Summary

### Key Concepts Covered

1. **Unity-ROS Integration**: Bridge architecture and communication protocols
2. **Robot Controllers**: Differential drive and manipulator control systems
3. **Sensor Simulation**: High-fidelity cameras, LiDAR, and depth sensors
4. **Advanced Rendering**: PBR materials, dynamic lighting, and environmental effects
5. **Interactive Training**: Module-based training systems with performance tracking
6. **Performance Optimization**: Efficient rendering and simulation techniques
7. **Real-world Applications**: Training, simulation, and visualization systems

### Practical Skills Acquired

- ✅ Build Unity-ROS 2 integration systems
- ✅ Create photorealistic robot simulations
- ✅ Implement advanced sensor models with noise and distortion
- ✅ Design interactive training applications
- ✅ Optimize rendering performance for complex scenarios

### Next Steps

This Unity foundation prepares you for **Chapter 9: Digital Twins**, where you'll explore how to create complete virtual replicas of physical robots and their environments. You'll learn how to:

- Build bidirectional synchronization between real and virtual systems
- Implement predictive maintenance and fault simulation
- Create real-time monitoring and control interfaces
- Develop advanced digital twin architectures

---

## 🤔 Chapter Reflection

1. **Visualization Impact**: How does Unity's rendering capability enhance robotics research and training compared to traditional simulation platforms?
2. **Integration Challenges**: What are the key considerations when connecting Unity simulations with real robot systems?
3. **Training Effectiveness**: How can interactive training applications improve robot operator skill development?
4. **Future Applications**: What emerging technologies (AR/VR, AI, cloud computing) could further enhance Unity-based robotics applications?

---

**[← Back to Quarter 2 Overview](index.md) | [Continue to Chapter 9: Digital Twins →](09-digital-twins.md)**