from visualizers.Visualizer import Visualizer


if __name__ == "__main__":
    visualizer = Visualizer("ck+", "../dataset_ck+", "png")

    # red: pose1 landmarks
    pose1 = "11-neutral"
    # blue: pose2 landmarks
    pose2 = "11-disgust"

    visualizer.compare(pose1, pose2)
