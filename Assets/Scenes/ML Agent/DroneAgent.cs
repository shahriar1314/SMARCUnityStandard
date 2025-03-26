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
using MathNet;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using Unity.Robotics.Core; //Clock
using Unity.Robotics.ROSTCPConnector;
using StdMessages = RosMessageTypes.Std;

using Random = UnityEngine.Random;

public class DroneAgent : Agent
{
    [Header("Basics")] 
    public GameObject BaseLink;
    public GameObject BaseLinkSAM;

    public Transform goal;             // where the drone should go 

    public DroneController.DroneController droneController;

    double massQuadrotor;
    
    public float targetHeight = 3f;  
    public float maxSpeed = 0.5f;     

    private ArticulationBody baseLinkDroneAB;
    private ArticulationBody[] ABparts;
    private Rigidbody[] RBparts;

    [Tooltip("Transform to Teleport(Drone Baselink)")]
    public Transform Target; // Transform to Teleport
    [Tooltip("Drone Actuator or Winch System (Rigid Body)")]
    public Transform DroneActuator;

    private Vector<double> initialPositionSAM;

    private Vector3 previousVelocity;

    private int immovableStage = 2;

    public override void Initialize()
    {
        if (goal == null)
        {
            Debug.LogWarning("Target or Goal not set for DroneAgent. Disabling.");
            enabled = false;
        }

        // Check if BaseLink exists and has an ArticulationBody component
        if (BaseLink == null)
        {
            Debug.LogError("BaseLink or ArticulationBody is missing!");
            enabled = false;
            return;
        }

        initialPositionSAM = BaseLinkSAM.transform.position.To<ENU>().ToDense();

        baseLinkDroneAB = BaseLink.GetComponent<ArticulationBody>();
        ABparts = Target.gameObject.GetComponentsInChildren<ArticulationBody>();
        RBparts = DroneActuator.gameObject.GetComponentsInChildren<Rigidbody>();
    }
    

    public override void CollectObservations(VectorSensor sensor)
    {   
        var position = BaseLink.transform.position;
        sensor.AddObservation(position.x);
        sensor.AddObservation(position.y);
        sensor.AddObservation(position.z);

        var velocity = baseLinkDroneAB.linearVelocity;
        sensor.AddObservation(velocity.x);
        sensor.AddObservation(velocity.y);
        sensor.AddObservation(velocity.z);

        var acceleration = (velocity - previousVelocity) / Time.fixedDeltaTime;
        previousVelocity = velocity; // Store for next frame
        sensor.AddObservation(acceleration.x);
        sensor.AddObservation(acceleration.y);
        sensor.AddObservation(acceleration.z);

        sensor.AddObservation(goal.position.x);
        sensor.AddObservation(goal.position.y);
        sensor.AddObservation(goal.position.z);

        Debug.Log($"[DroneAgent] Observations Collected | Pos : {position} | Vel: {velocity} | Acc: {acceleration}");
        Debug.Log($"[DroneAgent] Observations Collected | Goal: {goal.position}");
        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, goal.position);
        Debug.Log($"[DroneAgent] Observations Collected | DISTANCE TO GOAL: {distanceToGoal}");
        
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float velX = actionBuffers.ContinuousActions[0];
        float velY = actionBuffers.ContinuousActions[1];
        float velZ = actionBuffers.ContinuousActions[2];

        float accelX = actionBuffers.ContinuousActions[3];
        float accelY = actionBuffers.ContinuousActions[4];
        float accelZ = actionBuffers.ContinuousActions[5];

        Debug.Log($"[DroneAgent] ACTION RECEIVED | Vel: ({velX}, {velY}, {velZ}) | Acc: ({accelX}, {accelY}, {accelZ})");


        if (droneController != null)
        {
            droneController.TargetVelocity = new Vector3(velX, velY, velZ) * maxSpeed;
            droneController.TargetAccel = new Vector3(accelX, accelY, accelZ)*0;
        }

        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, goal.position);
        float velocityPenalty = -Vector3.Magnitude(baseLinkDroneAB.linearVelocity) * 0.1f;

        var reward = -distanceToGoal + velocityPenalty; 
        SetReward(reward);

        // Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} | Reward: {reward}");
    }

    public void FixedUpdate()
    {
        // Debug.Log("[DroneAgent] Method : E N T E R E D    I N T O    F I X E D U P D A T E");


        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, goal.position);
        // Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} ");

        if (distanceToGoal < 3f)
        {
            SetReward(2f);
            EndEpisode();
            Debug.Log($"EPISODE ENDED, GOAL REACHED, Distance to Goal:  {distanceToGoal}");
        }

        if (distanceToGoal > 35f)
        {
            SetReward(-1f);
            EndEpisode();
            Debug.Log($"EPISODE ENDED, DRONE IS TOO FAR, Distance to Goal: {distanceToGoal}");
        }
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("[DroneAgent] On Episode Beginning. Resetting agent...");

        for(int counter=1; counter<=200; counter++)
        {
            if(counter==1 || immovableStage!=2)
            {
                ResetPosition();
                switch(immovableStage)
                {
                    case 0:
                        immovableStage = 1;
                        break;
                    case 1:
                        if(Target.TryGetComponent(out ArticulationBody targetAb))
                        {
                            if(!targetAb.isRoot) return;
                            // targetAb.immovable = false;
                            Debug.Log("IMMOVABLE STAGE WORKED");
                        }
                        immovableStage = 2;
                        break; 
                    default:
                        break;
                }
                continue;
            }

            else
            {
                // Use the initial position from BaseLink and convert it properly
                var NewPosition = ENU.ConvertToRUF(
                    new Vector3(
                        (float)initialPositionSAM[0],  
                        (float)initialPositionSAM[1],
                        (float)initialPositionSAM[2]+5f//keeping the position same as the initial position of the drone 
                    ));
                // Use a default orientation (identity quaternion)
                var NewOrientation = Quaternion.identity;
                foreach (var ab in ABparts)
                {
                    ab.linearVelocity = Vector3.zero;
                    ab.angularVelocity = Vector3.zero;
                    ab.ResetArticulationBody();
                }


                for (int i = 0; i < RBparts.Length; i++)
                {   
                    // Reset position and rotation to initial values
                    RBparts[i].transform.position = NewPosition; // initialPositionWinch[i];
                    RBparts[i].rotation = NewOrientation;

                    // Reset velocity to stop movement
                    RBparts[i].linearVelocity = Vector3.zero;
                    RBparts[i].angularVelocity = Vector3.zero;
                } 
                continue;
            }
        }

        Debug.Log("******************* A G E N T   R E S E T ************************");
        Debug.Log($" [ResetAgent] Goal Position  : {goal.position.x}, {goal.position.y}, {goal.position.z}");
        Debug.Log($" [ResetAgent] Drone Position : {BaseLink.transform.position.x}, {BaseLink.transform.position.y}, {BaseLink.transform.position.z}");
        

    }


    void ResetPosition()
    {   
        
        // Use the initial position from BaseLink and convert it properly
        var NewPosition = ENU.ConvertToRUF(
            new Vector3(
                (float)initialPositionSAM[0],  
                (float)initialPositionSAM[1],
                (float)initialPositionSAM[2]+5f //keeping the position same as the initial position of the drone 
            ));
         // Use a default orientation (identity quaternion)
        var NewOrientation = Quaternion.identity;


        if (Target.TryGetComponent(out ArticulationBody targetAb))
        {
            if (!targetAb.isRoot) return;
            // targetAb.immovable = true;
            immovableStage = 0;
            targetAb.TeleportRoot(NewPosition, NewOrientation);
            targetAb.linearVelocity = Vector3.zero;
            targetAb.angularVelocity = Vector3.zero;
            targetAb.linearDamping = 0f;
        }
        else
        {
            Debug.Log("Target is not an Articulation Body");
        }


        foreach (var ab in ABparts)
        {
            ab.linearVelocity = Vector3.zero;
            ab.angularVelocity = Vector3.zero;
            ab.ResetArticulationBody();
        }


        for (int i = 0; i < RBparts.Length; i++)
        {   
            // Reset position and rotation to initial values
            RBparts[i].transform.position = NewPosition; // initialPositionWinch[i];
            RBparts[i].rotation = Quaternion.Euler(0f, 0f, 0f);

            // Reset velocity to stop movement
            RBparts[i].linearVelocity = Vector3.zero;
            RBparts[i].angularVelocity = Vector3.zero;
        }  

        Debug.Log("******************* A G E N T   I S    R E S E T ************************");          

    }
}