using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using VehicleComponents.Actuators;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using DefaultNamespace.LookUpTable;

using DefaultNamespace; // ResetArticulationBody() extension
using System.IO; // To logged the reward data 


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
using Unity.VisualScripting;

public class TargetAgent : Agent
{
    [Header("Basics")] 
    public GameObject BaseLink;
    public GameObject BaseLinkSAM;
    public Transform goal;             // where the drone should go 
    public DroneController.DroneController droneController;
    public UFO4 ufo;
    public float maxStep = 0.1f;   
    public bool debugMode = false;    

    private ArticulationBody baseLinkDroneAB;
    private ArticulationBody[] ABparts;
    private Rigidbody[] RBparts;

    [Tooltip("Transform to Teleport(Drone Baselink)")]
    public Transform Target; // Transform to Teleport
    [Tooltip("Drone Actuator or Winch System (Rigid Body)")]
    public Transform DroneActuator;

    private Vector<double> initialPositionSAM;
    private Vector3 elevatedGoalPos;

    private Vector3 previousVelocity;

    private int immovableStage = 0;
    private float cumulativeReward = 0f;
 

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

        elevatedGoalPos = new Vector3(
            goal.position.x,
            goal.position.y + 2f,
            goal.position.z
        );
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

        sensor.AddObservation(elevatedGoalPos.x);
        sensor.AddObservation(elevatedGoalPos.y);
        sensor.AddObservation(elevatedGoalPos.z);

        // Debug.Log($"[DroneAgent] Observations Collected | Drone Pos : {position} | Vel: {velocity} | Acc: {acceleration}");
        // Debug.Log($"[DroneAgent] Observations Collected | Goal: {goal.position}");
        // float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        // Debug.Log($"[DroneAgent] Observations Collected | DISTANCE TO GOAL: {distanceToGoal}");
        
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float posX = actionBuffers.ContinuousActions[0];
        float posY = actionBuffers.ContinuousActions[1];
        float posZ = actionBuffers.ContinuousActions[2];

        if(debugMode) Debug.Log($"[TargetAgent] ACTION RECEIVED : ({posX}, {posY}, {posZ})");

        var position = BaseLink.transform.position;

        if(ufo==null)
        {
            Debug.LogWarning("UFO not selected");
            return;
        }


        Vector3 newPosition = new Vector3(position.x + (posX * maxStep), position.y + (posY * maxStep), position.z+ (posZ * maxStep));
        ufo.SetPosition(newPosition);
        if(debugMode) Debug.Log($"[TargetAgent] NEW POSITION | Position: (X:{newPosition.x} Y:{newPosition.y} Z:{newPosition.z})");
    

        float maxDistance = 300f; // Maximum distance from the Drone to the goal 
        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        float normalizedDistance = distanceToGoal / maxDistance; // [0, 1]

        // float velocityPenalty = -Vector3.Magnitude(baseLinkDroneAB.linearVelocity) * 0.01f;
        float waterAvoidanceReward = (position.y > 0f && position.y < 2f) ? -10f : 0f;


        float reward = -normalizedDistance +waterAvoidanceReward; // + velocityPenalty;
        AddReward(reward);
        cumulativeReward += reward;

        // if(debugMode) Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} | Reward: {reward}");
    }

    public void FixedUpdate()
    {
        if(debugMode) Debug.Log("[DroneAgent] Method : E N T E R E D    I N T O    F I X E D U P D A T E");


        float distanceToGoal = Vector3.Distance(BaseLink.transform.position, elevatedGoalPos);
        if(debugMode) Debug.Log($"[DroneAgent] Distance to Goal: {distanceToGoal} ");
        var position = BaseLink.transform.position;

        if (distanceToGoal < 2f)
        {
            AddReward(10f);
            EndEpisode();
            Debug.Log($"EPISODE ENDED, *******GOAL REACHED*******, Distance to Goal:  {distanceToGoal}");
        }

        else if (distanceToGoal > 50f)
        {
            AddReward(-1000000f);
            EndEpisode();
            Debug.Log($"EPISODE ENDED, DRONE IS TOO FAR, Distance to Goal: {distanceToGoal}");
        }

        else if (position.y < 0f)
        {
            SetReward(-1000000f);
            EndEpisode();
            Debug.Log($"EPISODE ENDED, DRONE WENT UNDERWATER, Distance to Goal: {distanceToGoal}");
        }

    }

    public override void OnEpisodeBegin()
    {
        if(debugMode) Debug.Log("[DroneAgent] On Episode Beginning. Resetting agent...");
        
        immovableStage = 0;

        for(int counter=0; counter<3; counter++)
        {
            switch(immovableStage)
            {
                case 0:
                    ResetPosition();
                    immovableStage = 1;
                    break;
                case 1:
                    if(Target.TryGetComponent(out ArticulationBody targetAb))
                    {
                        if(!targetAb.isRoot) return;
                        // targetAb.immovable = false;
                        // Debug.Log("IMMOVABLE STAGE WORKED");
                    }
                    immovableStage = 2;
                    break; 
                default:
                    break;
            }     
        }

        if(debugMode) Debug.Log("******************* A G E N T   R E S E T ************************");
        if(debugMode) Debug.Log($" [ResetAgent] Goal Position  : {elevatedGoalPos.x}, {elevatedGoalPos.y}, {elevatedGoalPos.z}");
        if(debugMode) Debug.Log($" [ResetAgent] Drone Position : {BaseLink.transform.position.x}, {BaseLink.transform.position.y}, {BaseLink.transform.position.z}");
        

    }


    void ResetPosition()
    {   
        
        // Use the initial position from BaseLink and convert it properly
        var NewPosition = ENU.ConvertToRUF(
            new Vector3(
                (float)initialPositionSAM[0]+20f,  
                (float)initialPositionSAM[1]+5f,
                (float)initialPositionSAM[2]+7f //keeping the position same as the initial position of the drone 
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

        if(debugMode) Debug.Log("******************* A G E N T   I S    R E S E T T I N G ************************");          

    }
}