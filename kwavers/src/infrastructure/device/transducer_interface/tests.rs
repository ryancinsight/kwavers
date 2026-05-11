use super::*;

#[test]
fn test_communication_protocol_display() {
    assert_eq!(CommunicationProtocol::USB.to_string(), "USB");
    assert_eq!(CommunicationProtocol::Ethernet.to_string(), "Ethernet");
    assert_eq!(CommunicationProtocol::Mock.to_string(), "Mock");
}

#[test]
fn test_transducer_state_display() {
    assert_eq!(TransducerState::Idle.to_string(), "Idle");
    assert_eq!(TransducerState::Transmitting.to_string(), "Transmitting");
}

#[test]
fn test_mock_transducer_creation() {
    let transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    assert_eq!(transducer.state(), TransducerState::Idle);
    assert!(transducer.is_connected());
}

#[test]
fn test_mock_transducer_set_power() {
    let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    let response = transducer.send_command(HardwareCommand::SetPower(50.0));
    response.unwrap();
    assert!(transducer.current_power_percent - 50.0 < 0.01);
}

#[test]
fn test_mock_transducer_invalid_power() {
    let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    let response = transducer.send_command(HardwareCommand::SetPower(150.0));
    assert!(response.is_err());
    assert!(!transducer.last_error().unwrap().is_empty());
}

#[test]
fn test_mock_transducer_set_frequency() {
    let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    let response = transducer.send_command(HardwareCommand::SetFrequency(5.0e6));
    response.unwrap();
    assert!((transducer.current_frequency - 5.0e6).abs() < 1.0);
}

#[test]
fn test_device_manager_registration() {
    let mut manager = DeviceManager::new();
    let transducer = Box::new(MockTransducer::new(
        "TEST-1.5".to_string(),
        "TestCorp".to_string(),
    ));
    let result = manager.register_device("device_1".to_string(), transducer);
    result.unwrap();
    assert_eq!(manager.device_count(), 1);
}

#[test]
fn test_device_manager_duplicate_registration() {
    let mut manager = DeviceManager::new();
    let transducer1 = Box::new(MockTransducer::new(
        "TEST-1.5".to_string(),
        "TestCorp".to_string(),
    ));
    let transducer2 = Box::new(MockTransducer::new(
        "TEST-2.0".to_string(),
        "TestCorp".to_string(),
    ));

    let _ = manager.register_device("device_1".to_string(), transducer1);
    let result = manager.register_device("device_1".to_string(), transducer2);
    assert!(result.is_err());
}

#[test]
fn test_device_manager_list_devices() {
    let mut manager = DeviceManager::new();
    for i in 0..3 {
        let transducer = Box::new(MockTransducer::new(
            format!("TEST-{i}"),
            "TestCorp".to_string(),
        ));
        let _ = manager.register_device(format!("device_{i}"), transducer);
    }

    let devices = manager.list_devices();
    assert_eq!(devices.len(), 3);
}

#[test]
fn test_mock_transducer_calibration() {
    let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    let result = transducer.calibrate();
    result.unwrap();
    assert_eq!(transducer.state(), TransducerState::Idle);
}

#[test]
fn test_mock_transducer_telemetry() {
    let mut transducer = MockTransducer::new("TEST-1.5".to_string(), "TestCorp".to_string());
    let _ = transducer.send_command(HardwareCommand::SetPower(50.0));
    let telem = transducer.get_telemetry().unwrap();
    assert!(telem.measured_power_w > 0.0);
    assert!(telem.temperature_c > 20.0);
}

#[test]
fn test_device_specification_creation() {
    let spec = TransducerSpecification {
        model: "TEST-1.5".to_string(),
        manufacturer: "TestCorp".to_string(),
        serial_number: "SN-001".to_string(),
        frequency_range: (0.5e6, 10.0e6),
        max_power: 50.0,
        num_elements: 128,
        focal_length_mm: Some(60.0),
        element_diameter_mm: 0.3,
        calibration_date: "2026-01-30".to_string(),
        calibration_expiry: "2027-01-30".to_string(),
    };

    assert_eq!(spec.model, "TEST-1.5");
    assert_eq!(spec.num_elements, 128);
    assert_eq!(spec.focal_length_mm.unwrap(), 60.0);
}
