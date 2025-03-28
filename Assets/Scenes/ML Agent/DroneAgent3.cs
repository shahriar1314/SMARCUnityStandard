using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using VehicleComponents.Actuators;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using DefaultNamespace.LookUpTable;
using DefaultNamespace; // ResetArticulationBody() extension
using Force;
using Unity.Mathematics;
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Unity.Robotics.Core; // Clock
using Unity.Robotics.ROSTCPConnector;
using StdMessages = RosMessageTypes.Std;
using Random = UnityEngine.Random;

public class DroneAgent3 : Agent
{
    [Header("Basics")]
    public GameObject BaseLink;
    public GameObject BaseLinkSAM;
    public Transform goal;                     // Goal position
    public DroneController.DroneController droneController;

    public float maxSpeed = 0.75f;

    private ArticulationBody baseLinkDroneAB;
    private ArticulationBody[] ABparts;
    private Rigidbody[] RBparts;

    public Transform Target;
    public Transform DroneActuator;

    private Vector<double> initialPositionSAM;
    private Vector3 previousVelocity;
    private int immovableStage = 0;

    public bool debugMode = false;             // Toggle for debug logging

    public override void Initialize()
    {
        if (goal == null || BaseLink == null)
        {
            Debug.LogError("Goal or BaseLink not assigned!");
            enabled = false;
            return;
        }

        initialPositionSAM = BaseLinkSAM.transform.position.To<ENU>().ToDense();
        baseLinkDroneAB = BaseLink.GetComponent<ArticulationBody>();
        ABparts = Target.GetComponentsInChildren<ArticulationBody>();
        RBparts = DroneActuator.GetComponentsInChildren<Rigidbody>();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 pos = BaseLink.transform.position;
        Vector3 relPos = goal.position - pos;
        Vector3 velocity = baseLinkDroneAB.linearVelocity;
        Vector3 acceleration = (velocity - previousVelocity) / Time.fixedDeltaTime;
        previousVelocity = velocity;

        float distToGoal = relPos.magnitude;
        float accMag = acceleration.magnitude;
        float yaw = BaseLink.transform.eulerAngles.y / 360f; // Normalize yaw
        float timeNormalized = (float)(Time.fixedTime % 60f) / 60f;

        // Core observations
        sensor.AddObservation(relPos);
        sensor.AddObservation(velocity);
        sensor.AddObservation(distToGoal);
        sensor.AddObservation(accMag);
        sensor.AddObservation(yaw);
        sensor.AddObservation(timeNormalized); // Optional

        if (debugMode)
        {
            Debug.Log($"[Obs] relPos: {relPos}, vel: {velocity}, dist: {distToGoal}, yaw: {yaw}");
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Read 3D velocity command from actions
        float velX = actionBuffers.ContinuousActions[0];
        float velY = actionBuffers.ContinuousActions[1];
        float velZ = actionBuffers.ContinuousActions[2];

        if (droneController != null)
        {
            droneController.TargetVelocity = new Vector3(velX, velY, velZ) * maxSpeed;
        }

        Vector3 position = BaseLink.transform.position;
        Vector3 velocity = baseLinkDroneAB.linearVelocity;
        float distanceToGoal = Vector3.Distance(position, goal.position);

        // Reward shaping
        float distanceReward = -distanceToGoal * 0.1f;
        Vector3 directionToGoal = (goal.position - position).normalized;
        float alignment = Vector3.Dot(velocity.normalized, directionToGoal);
        float alignmentReward = alignment * 0.05f;
        float jerk = ((velocity - previousVelocity) / Time.fixedDeltaTime).magnitude;
        float jerkPenalty = -jerk * 0.01f;

        float totalReward = distanceReward + alignmentReward + jerkPenalty ;
        AddReward(totalReward);

        if (debugMode)
        {
            Debug.Log($"[Actions] Vel: ({velX}, {velY}, {velZ})");
            Debug.Log($"[Reward] Dist: {distanceToGoal}, Align: {alignment:F2}, Jerk: {jerk:F2}, Water: {position.y > 0f}");
            Debug.Log($"[Reward] Total: {totalReward}");
        }
    }

    public override void OnEpisodeBegin()
    {
        immovableStage = 0;

        for (int i = 0; i < 3; i++)
        {
            switch (immovableStage)
            {
                case 0:
                    ResetPosition();
                    immovableStage = 1;
                    break;
                case 1:
                    if (Target.TryGetComponent(out ArticulationBody ab) && ab.isRoot)
                        immovableStage = 2;
                    break;
            }
        }

        if (debugMode)
        {
            Debug.Log("*** EPISODE RESET ***");
            Debug.Log($"Goal: {goal.position}, Drone: {BaseLink.transform.position}");
        }
    }

    public void FixedUpdate()
    {
        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, goal.position);
        Vector3 position = BaseLink.transform.position;

        if (distanceToGoal < 1.5f)
        {
            SetReward(2f);
            EndEpisode();
            if (debugMode) Debug.Log("[Result] Goal Reached!!!!!!!!!!!!!!!!!");
        }
        else if (distanceToGoal > 30f)
        {
            SetReward(-1f);
            EndEpisode();
            if (debugMode) Debug.Log("[Result] Drone too far from goal.");
        }
        else if (position.y < 0f)
        {
            SetReward(-1f);
            EndEpisode();
            if(debugMode) Debug.Log("[Result] Drone fell into water.");
        }
    }

    void ResetPosition()
    {
        Vector3 newPosition = ENU.ConvertToRUF(new Vector3(
            (float)initialPositionSAM[0] + 5f,
            (float)initialPositionSAM[1],
            (float)initialPositionSAM[2] + 5f
        ));

        Quaternion newOrientation = Quaternion.identity;

        if (Target.TryGetComponent(out ArticulationBody targetAb) && targetAb.isRoot)
        {
            targetAb.TeleportRoot(newPosition, newOrientation);
            targetAb.linearVelocity = Vector3.zero;
            targetAb.angularVelocity = Vector3.zero;
            targetAb.linearDamping = 0f;
            immovableStage = 0;
        }

        foreach (var ab in ABparts)
        {
            ab.linearVelocity = Vector3.zero;
            ab.angularVelocity = Vector3.zero;
            ab.ResetArticulationBody();
        }

        foreach (var rb in RBparts)
        {
            rb.transform.position = newPosition;
            rb.rotation = Quaternion.Euler(0f, 0f, 0f);
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }
}
